"""Optimizer: Muon (hidden matrices) + Adam.

- ``MuonAdam``: Muon (Jordan et al. 2024; torch.optim.Muon) on the hidden weight matrices of the given MLPs, Adam on
  their first and last Linear and on all vectors. Muon's step is rescaled to the update RMS of Adam
  (adjust_lr_fn='match_rms_adamw', Liu et al. 2025), so the Adam learning rate is reused unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linears(m: nn.Module) -> list[nn.Linear]:
    return [x for x in m.modules() if isinstance(x, nn.Linear)]


class FlatAdam:
    """Adam (torch.optim.Adam's arithmetic, capturable: the learning rate is a 0-dim device tensor) on parameters that
    become views of one flat buffer, their gradients views of another, so a step is a few elementwise kernels over
    all elements. The fused multi-tensor Adam runs one thread block per 65536 elements: a handful of blocks for the
    critic, 0.27 ms per step on an RTX 4090 against ~0.03 ms here."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        params = [p for p in params if p.requires_grad]
        dev = params[0].device
        n = sum(p.numel() for p in params)
        self.flat = torch.empty(n, device=dev)
        self.gflat = torch.zeros(n, device=dev)
        o = 0
        for p in params:
            k = p.numel()
            self.flat[o : o + k].copy_(p.detach().flatten())
            p.data = self.flat[o : o + k].view_as(p)
            p.grad = self.gflat[o : o + k].view_as(p)
            o += k
        self.m, self.v = torch.zeros_like(self.flat), torch.zeros_like(self.flat)
        self.n_step = torch.zeros((), device=dev)
        self.betas, self.eps = betas, eps
        self.param_groups = [{"params": params, "lr": torch.tensor(float(lr), device=dev)}]

    def zero_grad(self, set_to_none: bool = True):
        self.gflat.zero_()  # the gradients stay views of gflat

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        self.n_step.add_(1)
        self.m.lerp_(self.gflat, 1 - b1)
        self.v.mul_(b2).addcmul_(self.gflat, self.gflat, value=1 - b2)
        bc1 = 1 - torch.pow(b1, self.n_step)
        bc2 = 1 - torch.pow(b2, self.n_step)
        denom = (self.v.sqrt() / bc2.sqrt()).add_(self.eps)
        self.flat.sub_(self.m / denom * (self.param_groups[0]["lr"] / bc1))

    def state_dict(self):
        return {"flat_adam": True, "m": self.m, "v": self.v, "step": self.n_step}

    def load_state_dict(self, sd: dict):
        assert sd.get("flat_adam"), "optimizer state of another Adam implementation"
        self.m.copy_(sd["m"]), self.v.copy_(sd["v"]), self.n_step.copy_(sd["step"])


class MuonAdam:
    """Muon on hidden matrices of ``mlps`` (all Linear layers except each MLP's first and last), Adam elsewhere.

    ``capturable`` (--cuda_graph): the learning rate is a 0-dim device tensor (set it with ``fill_``), Adam is the
    capturable fused kernel, and the Muon step is done here with the same arithmetic as torch.optim.Muon but with the
    tensor learning rate (torch.optim.Muon reads a tensor lr on the host, so a captured step would freeze it)."""

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True,
                 capturable: bool = False, io: bool = False, flat_adam: bool = False, weight_decay: float = 0.0):  # fmt: skip
        """``io``: the first and last Linear of each MLP on Muon too (Adam only on the vectors). ``flat_adam``: the Adam
        part as FlatAdam (capturable; no gated step). ``weight_decay``: decoupled decay on the Muon group
        (torch.optim.Muon default is 0.1)."""
        hidden = {id(l.weight) for m in mlps for l in (_linears(m) if io else _linears(m)[1:-1])}
        muon = [p for p in model.parameters() if id(p) in hidden]
        adam = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        self.capturable = capturable
        if capturable:
            dev = muon[0].device
            self.muon = torch.optim.Muon(muon, lr=torch.tensor(float(lr), device=dev), weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
            if flat_adam:
                self.adam = FlatAdam(adam, lr, betas=betas)
            else:
                self.adam = torch.optim.Adam(adam, lr=torch.tensor(float(lr), device=dev), betas=betas, fused=True, capturable=True)
        else:
            self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        # a property: Optimizer.load_state_dict replaces the group dicts, so a list built in __init__ goes stale on --resume
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none), self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for g in self.muon.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.muon.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                buf = st["momentum_buffer"]
                if ok is None:
                    buf.lerp_(p.grad, 1 - g["momentum"])
                else:
                    buf.copy_(torch.where(ok, buf.lerp(p.grad, 1 - g["momentum"]), buf))
                update = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                update = _zeropower_via_newtonschulz(update, g["ns_coefficients"], g["ns_steps"], g["eps"])
                ratio = _adjust_lr(1.0, g["adjust_lr_fn"], p.shape)
                d = update.to(p.dtype) * (g["lr"] * -ratio)  # fp32 product (a bf16 update x 0-dim tensor stays bf16)
                if g["weight_decay"] != 0.0:  # decoupled decay as torch.optim.Muon: p *= 1 - lr * wd, then the update
                    shrink = 1 - g["lr"] * g["weight_decay"]
                    p.mul_(shrink if ok is None else torch.where(ok, shrink, 1.0))
                p.add_(d if ok is None else torch.where(ok, d, 0.0))

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        """``ok``: optional 0-dim bool device tensor; where it is false the whole step is skipped without a host read
        (parameters, momenta, Adam moments and step counts unchanged). The gradients must be finite (zero them first)."""
        if ok is not None:
            if self.capturable:
                self._muon_step_tensor_lr(ok)
            else:  # torch.optim.Muon's own step (unchanged arithmetic), undone where ok is false
                ps = [p for g in self.muon.param_groups for p in g["params"] if p.grad is not None]
                bufs = lambda: [self.muon.state[p]["momentum_buffer"] for p in ps]  # noqa: E731
                for p in ps:
                    self.muon.state[p].setdefault("momentum_buffer", torch.zeros_like(p.grad, memory_format=torch.preserve_format))
                keep = [t.clone() for t in ps + bufs()]
                self.muon.step()
                for t, k in zip(ps + bufs(), keep):
                    t.copy_(torch.where(ok, t, k))
            assert self.adam.defaults["fused"], "a gated step needs the fused Adam (--fused_adam)"
            self.adam.found_inf = (~ok).float()  # the fused kernel skips the step and undoes the step count
            try:
                self.adam.step()
            finally:
                self.adam.found_inf = None
            return
        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd: dict):
        lrs = [g["lr"] for g in self.param_groups]
        self.muon.load_state_dict(sd["muon"]), self.adam.load_state_dict(sd["adam"])
        if self.capturable:  # keep the device learning-rate tensors (the training loop fills them)
            for g, lr in zip(self.param_groups, lrs):
                lr.fill_(float(g["lr"]))
                g["lr"] = lr


def check():
    """Unit checks: MuonAdam puts exactly the hidden matrices on Muon; the capturable step (also inside a CUDA graph)
    matches the plain one; the gated step (``ok``) skips a non-finite update without changing any state."""
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. MuonAdam split
    from partymimic.networks import Policy

    pi = Policy(20, 30, 5, hist=3, enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16))
    oa = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-3, fused=False)
    n_muon = sum(len(g["params"]) for g in oa.muon.param_groups)
    n_adam = sum(len(g["params"]) for g in oa.adam.param_groups)
    chk("MuonAdam: hidden matrices on Muon (2 per MLP), the rest on Adam", n_muon == 4 and n_muon + n_adam == len(list(pi.parameters())),
        f"muon {n_muon}, adam {n_adam}")  # fmt: skip
    # 2. capturable MuonAdam (tensor lr, own Muon step, capturable fused Adam) == MuonAdam, eagerly and inside a CUDA graph.
    #    Muon orthogonalizes in bf16, so a 1e-7 difference flips bf16 roundings after a few steps; the yardstick is the
    #    spread of plain MuonAdam itself under a 1e-6 relative lr perturbation.
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        pis = [Policy(20, 30, 5, hist=3, enc_hidden=(64, 48, 16), dec_hidden=(64, 32, 16)).to(dev) for _ in range(4)]
        for q in pis[1:]:
            q.load_state_dict(pis[0].state_dict())
        init = [p.detach().clone() for p in pis[0].parameters()]
        mk = lambda q, cap: MuonAdam(q, [q.enc, q.dec], lr=1e-3, capturable=cap)  # noqa: E731
        opts = [mk(pis[0], False), mk(pis[1], True), mk(pis[2], True), mk(pis[3], False)]
        xo, xg = torch.randn(256, 3, 20, device=dev), torch.randn(256, 30, device=dev)
        lrs = [1e-3 * (1 - 0.02 * i) for i in range(20)]

        def stepf(k):
            opts[k].zero_grad(set_to_none=True)
            (pis[k](xo, xg) ** 2).mean().backward()
            opts[k].step()

        graph = None
        for i, lr in enumerate(lrs):
            for k in range(4):
                lr_k = lr * (1 + 1e-6) if k == 3 else lr
                for gr in opts[k].param_groups:
                    gr["lr"].fill_(lr_k) if torch.is_tensor(gr["lr"]) else gr.__setitem__("lr", lr_k)
            stepf(0), stepf(1), stepf(3)
            if i < 3:  # warm-up on a side stream, then capture on the next step and replay
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stepf(2)
                torch.cuda.current_stream().wait_stream(side)
            else:
                if graph is None:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        stepf(2)
                graph.replay()

        def rel(k):  # max over tensors of |p_0 - p_k| / |total change of p_0|
            return max(float((p.detach() - q.detach()).norm() / (p.detach() - p0).norm().clamp(min=1e-12))
                       for p, q, p0 in zip(pis[0].parameters(), pis[k].parameters(), init))  # fmt: skip

        floor = rel(3)
        chk("capturable MuonAdam == MuonAdam (20 steps, changing lr)", rel(1) < max(3 * floor, 1e-4),
            f"max diff / total change {rel(1):.1e} (noise floor {floor:.1e})")  # fmt: skip
        chk("capturable MuonAdam in a CUDA graph == MuonAdam", rel(2) < max(3 * floor, 1e-4), f"{rel(2):.1e}")
        # 3. gated step (train.py --no_sync skips a non-finite policy update on the GPU): ok=True is bitwise the plain
        #    step, ok=False changes neither the parameters nor any optimizer state (also as a graph replay)
        import copy

        def mk_net():
            m = nn.Module()
            m.enc = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))
            return m.to(dev)

        def snap(m, o):
            t = [p.detach().clone() for p in m.parameters()]
            for sub in (o.muon, o.adam):
                for st in sub.state.values():
                    t += [v.detach().clone() for v in st.values() if torch.is_tensor(v)]
            return t

        def set_grads(m, seed):
            g = torch.Generator(device=dev).manual_seed(seed)
            for p in m.parameters():
                p.grad = torch.randn(p.shape, generator=g, device=dev)

        same = lambda xs, ys: all(torch.equal(x, y) for x, y in zip(xs, ys))  # noqa: E731
        for cap in (True, False):
            a = mk_net()
            b = copy.deepcopy(a)
            oa = MuonAdam(a, [a.enc], lr=1e-3, capturable=cap)
            ob = MuonAdam(b, [b.enc], lr=1e-3, capturable=cap)
            for i in range(3):
                set_grads(a, i), set_grads(b, i)
                oa.step(), ob.step(ok=torch.tensor(True, device=dev))
            chk(f"gated step ok=True == plain step bitwise (capturable={cap})", same(snap(a, oa), snap(b, ob)))
            before = snap(b, ob)
            set_grads(b, 99)
            ob.step(ok=torch.tensor(False, device=dev))
            chk(f"gated step ok=False leaves parameters and state unchanged (capturable={cap})", same(before, snap(b, ob)))
        m = mk_net()
        o = MuonAdam(m, [m.enc], lr=1e-3, capturable=True)
        okt = torch.ones((), dtype=torch.bool, device=dev)
        for i in range(3):
            set_grads(m, i), o.step(ok=okt)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(3):
                set_grads(m, 10 + i), o.step(ok=okt)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o.step(ok=okt)
        before = snap(m, o)
        okt.fill_(False), graph.replay(), torch.cuda.synchronize()
        frozen = same(before, snap(m, o))
        okt.fill_(True), graph.replay(), torch.cuda.synchronize()
        chk("gated step in a CUDA graph: ok=False unchanged, ok=True moves", frozen and not same(before, snap(m, o)))
        # 4. FlatAdam == torch.optim.Adam (eagerly and as a graph replay, changing lr)
        ref, fa, fg = mk_net(), mk_net(), mk_net()
        fa.load_state_dict(ref.state_dict()), fg.load_state_dict(ref.state_dict())
        o_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
        o_fa, o_fg = FlatAdam(fa.parameters(), 1e-3), FlatAdam(fg.parameters(), 1e-3)
        x = torch.randn(512, 8, device=dev)
        graph = None
        for i in range(20):
            lr = 1e-3 * (1 - 0.03 * i)
            o_ref.param_groups[0]["lr"] = lr
            o_fa.param_groups[0]["lr"].fill_(lr), o_fg.param_groups[0]["lr"].fill_(lr)
            for m_, o_ in ((ref, o_ref), (fa, o_fa)):
                o_.zero_grad(set_to_none=True)
                ((m_.enc(x) - 1) ** 2).mean().backward()
                o_.step()
            if graph is None and i >= 3:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    o_fg.zero_grad()
                    ((fg.enc(x) - 1) ** 2).mean().backward()
                    o_fg.step()
            if graph is not None:
                graph.replay()
            else:
                o_fg.zero_grad()
                ((fg.enc(x) - 1) ** 2).mean().backward()
                o_fg.step()
        err = max(float((p - q).abs().max()) for p, q in zip(ref.parameters(), fa.parameters()))
        err_g = max(float((p - q).abs().max()) for p, q in zip(ref.parameters(), fg.parameters()))
        chk("FlatAdam == torch.optim.Adam (20 steps, changing lr; eager / CUDA graph)", err < 1e-5 and err_g < 1e-5,
            f"max |diff| {err:.1e} / {err_g:.1e}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
