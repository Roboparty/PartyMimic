"""World model, policy (motion encoder + shared decoder), and critic (SuperMimic paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class FrozenMLP(nn.Module):
    """A trained ELU MLP with its input standardization, held as buffers: no optimizer, EMA or --pi_opt grouping of the
    module that owns it touches it, and it is saved and exported with that module's state_dict."""

    def __init__(self, inp: int, hidden: list[int], out: int):
        super().__init__()
        dims = [inp, *hidden, out]
        self.n = len(dims) - 1
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            self.register_buffer(f"w{i}", torch.zeros(b, a))
            self.register_buffer(f"b{i}", torch.zeros(b))
        self.register_buffer("mu", torch.zeros(inp))
        self.register_buffer("sd", torch.ones(inp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sd
        for i in range(self.n):
            x = nn.functional.linear(x, getattr(self, f"w{i}"), getattr(self, f"b{i}"))
            if i < self.n - 1:
                x = nn.functional.elu(x)
        return x

    @torch.no_grad()
    def load_from(self, net: nn.Sequential, mu: torch.Tensor, sd: torch.Tensor) -> None:
        """Copy the Linear layers of ``mlp(inp, hidden, out)`` (trained on (x - mu) / sd)."""
        lin = [m for m in net if isinstance(m, nn.Linear)]
        assert len(lin) == self.n
        for i, m in enumerate(lin):
            getattr(self, f"w{i}").copy_(m.weight)
            getattr(self, f"b{i}").copy_(m.bias)
        self.mu.copy_(mu), self.sd.copy_(sd)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    # InstantTrack: when true, the freeze test runs on the GPU (torch.where) instead of a host-side `if`
    no_sync: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if not RunningNorm.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        if RunningNorm.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
        else:
            self.mean.copy_(mean)
            self.var = var
            self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``use_prev_action`` the model also sees a_{t-1}: delayed actuators still track the previous
    action's target during the first physics steps of a control step.
    """

    def __init__(self, num_j: int, xi_dim: int, use_prev_action: bool = False, hidden=(2048, 2048, 2048),
                 contact_modes: bool = False, structured: bool = False, dt: float = 0.02):  # fmt: skip
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off."""
        super().__init__()
        self.num_j = num_j
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        # ``structured`` = semi-implicit integration (positions follow the predicted velocities,
        # the net predicts velocity increments plus small position corrections)
        self.structured, self.dt = structured, dt
        self.in_dim = inv_dim(num_j) + num_j * (2 if use_prev_action else 1) + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        n_out = 4 * self.out_dim + 2 if contact_modes else self.out_dim
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # scale of the plain frame-local increments, used to normalize the training loss (equal to out_norm
        # unless ``structured`` changes what the network outputs)
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, S, a, a_prev, xi_n):
        if self.use_prev_action:
            return torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    # weight of the new velocity in the position drift: n semi-implicit substeps with a linearly changing
    # velocity give sum_k k/n / n = (n + 1) / (2 n); n = 4 substeps -> 0.625 (measured residual std of the
    # joint-position increment: 16 % with 0.5, 11 % with 0.625, 17 % with 1.0)
    DRIFT_W = 0.625

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the
        increments: yaw frame for the root position, body frame for the rotation vector."""
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
        return (mv3(H.transpose(-1, -2), a * v + b * v1), mv3(R.transpose(-1, -2), a * w + b * w1), a * qd + b * qd1)

    def targets(self, S, S1):
        """Regression targets of the network: the frame-local increments, or with ``structured`` the
        velocity increments plus the position corrections on top of the semi-implicit drift."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, J, J), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv3(H.transpose(-1, -2), p1 - p),
                so3_log(mm3(Rt, R1)),
                mv3(Rt, v1 - v),
                mv3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, a_prev, xi_n):
        return self.step_c(S, a, a_prev, xi_n)[0]

    def step_c(self, S, a, a_prev, xi_n):
        """Next state and next-step foot-contact logits (B, 2), or None without ``contact_modes``."""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, a_prev, xi_n)))
        logits = None
        if self.contact_modes:
            logits = y[..., -2:]
            cl, cr = torch.sigmoid(logits).unbind(-1)
            mode_w = torch.stack(((1 - cl) * (1 - cr), cl * (1 - cr), (1 - cl) * cr, cl * cr), -1)
            y = (y[..., :-2].unflatten(-1, (4, self.out_dim)) * mode_w.unsqueeze(-1)).sum(-2)
        return self.integrate(S, y * self.out_norm.std() + self.out_norm.mean), logits

    def integrate(self, S, d):
        """Next state from the denormalized network output ``d`` (the layout of ``targets``)."""
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        qd1 = qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        p1 = p + mv3(H, dp)
        R1 = mm3(R, so3_exp(dth))
        return (p1, R1, v1, w1, q + dq, qd1)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv3(H.transpose(-1, -2), ph - p),
                so3_log(mm3(Rt, Rh)),
                mv3(Rt, vh - v),
                mv3(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class _MHA(nn.Module):
    """Multi-head attention from Linear projections (Muon sees them as matrices) and scaled_dot_product_attention."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, y, causal: bool = False):
        B, n, d = x.shape
        sp = lambda t: t.view(B, -1, self.h, d // self.h).transpose(1, 2)  # noqa: E731
        out = nn.functional.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(y)), sp(self.v(y)), is_causal=causal)
        return self.o(out.transpose(1, 2).reshape(B, n, d))


def _sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / 10000 ** (i / d)
    return torch.stack((ang.sin(), ang.cos()), -1).flatten(1)


class XAttnEncoder(nn.Module):
    """RGMT's dynamics-conditioned command encoder: a causal Transformer block over the (normalized) observation history
    gives the dynamics embedding h_t (its newest token); the query MLP(h_t) attends to the goal frames' tokens (MLP +
    sinusoidal position) in one pre-LN cross-attention block; u_t = LN(...) in R^d."""

    def __init__(self, obs_dim: int, frame_dim: int, K: int, hist: int, d: int = 128, heads: int = 4):
        super().__init__()
        self.K = K
        self.obs_in = mlp(obs_dim, [d], d, act=nn.SiLU)
        self.register_buffer("pos_h", _sinusoid(hist, d))
        self.ln_h1, self.sa, self.ln_h2, self.ff_h = nn.LayerNorm(d), _MHA(d, heads), nn.LayerNorm(d), mlp(d, [2 * d], d, act=nn.SiLU)
        self.q_mlp = mlp(d, [d], d, act=nn.SiLU)
        self.cmd = mlp(frame_dim, [d], d, act=nn.SiLU)
        self.register_buffer("pos_c", _sinusoid(K, d))
        self.ln_q, self.xa, self.ln_s, self.ff = nn.LayerNorm(d), _MHA(d, heads), nn.LayerNorm(d), mlp(d, [2 * d], d, act=nn.SiLU)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, o_n: torch.Tensor, g_n: torch.Tensor) -> torch.Tensor:
        x = self.obs_in(o_n) + self.pos_h
        x = x + self.sa(self.ln_h1(x), self.ln_h1(x), causal=True)
        x = x + self.ff_h(self.ln_h2(x))
        q = self.q_mlp(x[:, -1:])  # (B, 1, d)
        Z = self.cmd(g_n.unflatten(-1, (self.K, -1))) + self.pos_c
        s = q + self.xa(self.ln_q(q), Z)
        s = s + self.ff(self.ln_s(s))
        return self.ln_out(s).squeeze(1)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh. ``ref_enc`` = "xattn": the latent is
    XAttnEncoder(o_{t-9:t}, g) in R^xattn_dim instead (it depends on the state, so ``encode`` does not exist).

    ``ref_skip`` = (offset, q0, alpha) (--pi_ref_skip): reference-action skip connection,
    a = (q_ref(f+1) - q0) / alpha + dec(...), with q_ref(f+1) read from the raw (unnormalized) goal vector at
    ``offset``; the network only learns the correction to tracking the reference with the PD controller.

    ``obs_err`` > 0 (--interface sparse5): the last obs_err entries of every observation are the keypoint error block
    (mdp.MDP.kp_err); the reference encoder does not see it and the decoder gets it of the newest step only.

    ``kp_skip`` = (alpha, hidden) (--kp_skip, sparse5): the reference-action skip with an estimated reference,
    a = kp2q(g) / alpha + dec(...), kp2q a FrozenMLP from the raw goal to q_ref(f+1) - q0, fit to the motion library
    before training. ``kp_aux`` > 0 (--kp_aux): a linear head ``aux`` on the reference latent (``latent``) with that
    many outputs, the auxiliary prediction of the goal frames' joint positions."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_skip=None,
                 ref_enc: str = "mlp", goal_frames: int = 10, xattn_dim: int = 128, xattn_heads: int = 4,
                 obs_err: int = 0, kp_skip=None, kp_aux: int = 0):  # fmt: skip
        super().__init__()
        self.hist, self.obs_err = hist, obs_err
        base = obs_dim - obs_err
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer("ref_q0", torch.tensor(ref_skip[1], dtype=torch.float32))
            self.register_buffer("ref_alpha", torch.tensor(ref_skip[2], dtype=torch.float32))
        self.xattn = ref_enc == "xattn"
        if self.xattn:
            z_dim = xattn_dim
            self.enc = XAttnEncoder(base, goal_dim // goal_frames, goal_frames, hist, xattn_dim, xattn_heads)
        else:
            self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + base * hist + obs_err, list(dec_hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)
        self.kp2q = None
        if kp_skip is not None:
            self.kp2q = FrozenMLP(goal_dim, list(kp_skip[1]), act_dim)
            self.register_buffer("kp_alpha", torch.tensor(kp_skip[0], dtype=torch.float32))
        self.aux = nn.Linear(z_dim, kp_aux) if kp_aux else None

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last. ``z`` (optional) is
        ``encode(goal)`` computed beforehand."""
        o_n = self.obs_norm(obs_hist)
        o = o_n.flatten(-2)
        if self.obs_err:
            o = torch.cat((o_n[..., : -self.obs_err].flatten(-2), o_n[..., -1, -self.obs_err :]), -1)
            o_n = o_n[..., : -self.obs_err]
        if self.xattn:
            z = run_mlp(lambda x: self.enc(x, self.goal_norm(goal)), o_n)
        elif z is None:
            z = self.encode(goal)
        a = run_mlp(self.dec, torch.cat((z, o), -1))
        if self.ref_off is not None:
            q_ref = goal[..., self.ref_off : self.ref_off + a.shape[-1]]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        if self.kp2q is not None:
            a = a + self.kp2q(goal) / self.kp_alpha
        return a

    def latent(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """The reference latent the decoder gets (``encode(goal)``, or the cross-attention output on obs_hist)."""
        if not self.xattn:
            return self.encode(goal)
        o_n = self.obs_norm(obs_hist)
        if self.obs_err:
            o_n = o_n[..., : -self.obs_err]
        return run_mlp(lambda x: self.enc(x, self.goal_norm(goal)), o_n)

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))


# Switches of the development code that this release does not have, with their "off" value. Run arguments (args.json,
# the "args" of a checkpoint) of that code still carry them; at the off value they are ignored, any other value
# describes a run that this code cannot rebuild or continue.
PRUNED_OFF = {
    "actor_lam": 1.0, "grad_decay": 1.0, "anchored": False, "demo_replay": False,
    "imag_action_noise": 0.0, "wm_prio": 0.0, "imag_prio": 0.0, "critic_alive": False, "critic_prior": False,
    "critic_grad": 0.0, "critic_ema": 0.0, "lookahead_k": 0, "wm_batches_early": 0, "wm_foot_weight": 0.0,
    "bc_sharp_weight": 0.0, "pi_grad_batch": 0, "no_critic": False, "wm_imag_ema": 0.0, "anchor_coef": 0.0,
    "boot_coef": 1.0, "grad_clip_step": 0.0, "det_frac": 0.0, "wm_load": None, "wm_freeze": False, "wm_ready": 0.0,
    # experiments of the BlinkMimic / PartyMimic development that the final recipe does not use
    "eprobe_share": 0.0, "act_cot_clip": 0.0, "action_target": None, "real_v": 0.0, "real_td": 0.0, "sil_multi": False,
    "samp_fix": False, "fail_fix": False, "bin_debias": False, "start_recent": 0.0, "wide_mix": 0.0, "critic_ens": 1,
    "vgrad_w": 1.0, "v_smooth": 0.0, "rise_focus": 0.0, "rise_relax": 0.0, "wm_ground_w": 0.0, "bin_fix": True,
    "flight_relax": 0.0, "feas_relax": 0.0, "cmd_noise": 0.0, "fall_share": 0.0, "pace_hard": None, "contact_trunc": False,
    "mpc_teacher": 0.0, "state_gnorm": 0, "critic_hidden": "256,128", "v_hi": 0.0, "v_target": 0.0, "v_huber": 0.0,
    # these also change the networks (inputs or architecture)
    "wm_pd": False, "pi_arch": "mlp", "pi_fourier": None, "pi_goal_feet": False, "wm_contact": 0.0, "heading": "x",
    "goal_offsets": "", "id_ff": 0.0, "safe_frames": False,
}  # fmt: skip
PRUNED_NET = ("wm_pd", "pi_arch", "pi_fourier", "pi_goal_feet", "wm_contact", "heading", "goal_offsets", "id_ff",
              "safe_frames")  # fmt: skip


def check_pruned(a_: dict, nets_only: bool = False, what: str = "this run") -> None:
    """Raise if the run arguments ``a_`` use a removed switch. ``nets_only``: only the switches that change the
    networks (enough to rebuild and evaluate a policy / world model); otherwise also every removed training switch
    (for tools that recompute the training objective)."""
    keys = PRUNED_NET if nets_only else PRUNED_OFF
    bad = {k: a_[k] for k in keys if k in a_ and a_[k] != PRUNED_OFF[k] and not (PRUNED_OFF[k] is None and not a_[k])}
    if bad:
        raise ValueError(f"{what} uses switches that this code does not have: {bad}")


def policy_kwargs(a_: dict, robot) -> dict:
    """Policy keyword arguments of a run with the training arguments ``a_``."""
    check_pruned(a_, nets_only=True)
    kw = {"ref_skip": policy_ref_skip(a_, robot)}
    if a_.get("ref_enc", "mlp") != "mlp":
        kw.update(ref_enc=a_["ref_enc"], goal_frames=a_["goal_frames"], xattn_dim=a_.get("xattn_dim", 128),
                  xattn_heads=a_.get("xattn_heads", 4))
    if a_.get("interface", "joint") != "joint":
        from .mdp import obs_err_dim

        kw["obs_err"] = obs_err_dim(a_["interface"])
    if a_.get("kp_skip", False):
        kw["kp_skip"] = (robot.action_scale_list(), tuple(int(x) for x in a_["kp_hidden"].split(",")))
    if a_.get("kp_aux", 0.0) > 0:
        kw["kp_aux"] = a_["goal_frames"] * len(robot.default_joint_pos_list())
    return kw


def policy_ref_skip(a_: dict, robot):
    """Policy(ref_skip=...) argument of a run with the training arguments ``a_`` (None without --pi_ref_skip, and with
    --interface sparse5, whose goal has no joint positions: the action base is the default pose q0)."""
    if not a_.get("pi_ref_skip", False) or a_.get("interface", "joint") != "joint":
        return None
    from .motion import GOAL_Q_OFFSET

    return (GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list())


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
