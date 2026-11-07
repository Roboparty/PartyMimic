# PartyMimic

**General humanoid motion tracking in minutes on one GPU.** PartyMimic trains one whole-body tracking policy for a
9,173-clip motion library (21 h) in **7–30 minutes on a single RTX 4090**. On held-out motions it has the lowest
tracking error of every method we compared, on two humanoids (Unitree G1 and RoboParty RP1).

PartyMimic is a model-based learner in the SuperMimic family. It fits an online MLP world model, trains the
policy by backpropagating through 32 imagined steps of that model, and uses a critic beyond the horizon. Its
training recipe is built for generalization and speed. Project page: [`docs/index.html`](docs/index.html), with
videos, the recipe ablation and the SuperMimic full-factor ablation.

![Loc/Man error against training compute](docs/static/hero.png)

## Results

Body position error (mm, lower is better) on the held-out MotionDecode splits, scored by the
[sim2real leaderboard](https://egalahad.github.io/sim2real/leaderboard/) sim2sim evaluator. Loc/Man is the mean of
Loc and Man.

| G1 | Training | Loc/Man | Loc | Man | Dance | Ground |
|---|---:|---:|---:|---:|---:|---:|
| SONIC (official) | 21,000 GPU-h | 37.1 | 28.7 | 45.5 | 48.7 | 80.3 |
| Mimic Lite v1.1 (official) | 50 GPU-h | 26.7 | 25.1 | 28.2 | 43.1 | 144.7 |
| SuperMimic | 52 min | 40.2 | 40.5 | 39.8 | 64.3 | 167.8 |
| NablaMimic | 99 min | 29.8 | 32.2 | 27.3 | 46.8 | 110.6 |
| BlinkMimic | 59 min | 31.0 | 31.9 | 30.1 | 53.8 | 98.7 |
| **PartyMimic** (2k iterations) | **6.7 min** | 26.2 | 29.3 | 23.1 | 43.1 | 84.2 |
| **PartyMimic** (12k iterations) | **30 min** | **23.6** | 25.5 | **21.6** | **38.1** | **74.3** |

| RP1 | Training | Loc/Man | Loc | Man | Dance | Ground |
|---|---:|---:|---:|---:|---:|---:|
| SuperMimic | 149 min | 35.0 | 39.9 | 30.2 | 56.4 | 158.4 |
| BlinkMimic | 68 min | 30.7 | 32.2 | 29.1 | 51.7 | 126.0 |
| **PartyMimic** (2k iterations) | **8.0 min** | 21.2 | 24.6 | 17.8 | 39.9 | 115.7 |
| **PartyMimic** (12k iterations) | **31 min** | **20.0** | **23.6** | **16.5** | **35.9** | **103.6** |

<sub>SuperMimic (paper hyperparameters), NablaMimic (official config), BlinkMimic (main preset) and PartyMimic were
all trained by us. Each run had one RTX 4090 to itself, and all used the same 9,173 clips (BONES-SEED + LAFAN1,
retargeted per robot). Time is training wall-clock on one GPU. PartyMimic rows are 2-seed means.
Test motions come from a different mocap source; no 1-s window lies within 0.059 rad RMS of the training set.
SONIC and Mimic Lite are official leaderboard entries with different data and hardware. Full protocol and
completion rates are on the [project page](docs/index.html#results).</sub>

What matters most in the recipe, measured by the effect of removing each factor on G1 Loc/Man:

| Factor | Effect when removed |
|---|---|
| Muon for policy and world model | Adam instead: +10.7 mm |
| Narrow tracking-reward kernels (`--std_scale 0.35`) | +6.5 mm |
| 32-step imagination | 16 steps: +5.9 mm |
| Reference-action skip | +5.6 mm |
| Policy EMA | +3.4 mm |
| Value-gradient consistency, Muon on all layers, diversity-balanced resets, self-imitation | 0.5–1.6 mm each; Ground +6–19 mm without balanced resets |

The page explains why each factor matters.

## Setup

Linux x86_64, an NVIDIA GPU with a CUDA 12.8 driver, and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> partymimic && cd partymimic
uv sync                    # mjlab, MuJoCo Warp, torch 2.9.0 (CUDA 12.8)
tools/download_data.sh     # G1 description + 40 G1-retargeted LAFAN1 clips -> data/ (enough for a first run)
```

RP1 is not redistributed here. Put its capsule-collision MJCF and meshes (`rp1_28dof_mjlab`) in
`data/rp1_28dof_mjlab/` and its motions in `data/rp1_motions/`, or pass `--motion_dir`.

**Motion data.** `--motion_dir` takes a directory of `.npz` clips, each with `fps` (50), `joint_pos` `[T, J]`,
`body_pos_w` `[T, B, 3]` and `body_quat_w` `[T, B, 4]` (wxyz). The joints and bodies follow the robot's order
(`src/partymimic/<robot>/`), with the pelvis first.
The results above used 10,000 randomly sampled BONES-SEED G1 retargets. Clips with a body more than 3 cm below the
ground or a joint speed above 25 rad/s were removed, leaving 9,133, plus the 40 LAFAN1 clips. For RP1, the same
motions were retargeted to RP1.

## Train

The defaults of `partymimic.train` are the PartyMimic recipe:

```bash
bash scripts/train.sh logs/g1_s0  --robot g1mj --motion_dir <npz clips> --seed 0   # 12k iterations, ~30 min on a 4090
bash scripts/train.sh logs/rp1_s0 --robot rp1  --motion_dir <npz clips> --seed 0
```

Checkpoints are written every 2,000 iterations (`ckpt_002000.pt`, ...). On G1, Loc/Man already beats the
previous best at 2,000 iterations. At 8,000 iterations Loc/Man, Dance and Ground are all the best in the comparison.
`uv run python -m partymimic.train -h` lists the remaining options.

## Evaluate, export, visualize

```bash
# sim2real leaderboard protocol (ONNX + deploy YAML, MuJoCo CPU sim2sim on MotionDecode)
SIM2REAL=<checkout of github.com/EGalahad/sim2real> bash scripts/eval.sh logs/g1_s0/ckpt_012000.pt eval/g1_s0 g1
# side-by-side videos of the median clip per split: robot solid, reference ghost, both at global poses
MUJOCO_GL=egl python tools/sim2sim_video.py eval/g1_s0 g1 <robot MJCF> videos --sim2real <sim2real checkout>
```

Also included: `tools/export_deploy_onnx.py`, which writes the ONNX and deploy YAML for the sim2real controller
(sim2sim and real robot). `tools/eval_batch.py` and `tools/eval_metrics.py` give fast in-simulator evaluation
on the training clips. `tools/export_onnx.py` and `tools/play_onnx.py` produce a single-motion ONNX in the
BeyondMimic layout.

## Attribution and license

PartyMimic is derived from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic), which builds on
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) and
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack). All three are MIT-licensed; their notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Methods: SuperMimic (Liao et al., arXiv:2610.28480, 2026), SHAC (Xu et al., 2022), BeyondMimic (Liao et al., 2025)
and Muon (Jordan et al., 2024).
Simulation: [mjlab](https://github.com/mujocolab/mjlab) and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
Code: MIT ([LICENSE](LICENSE)). Motion data, videos and checkpoints are derived from third-party motion datasets
(LAFAN1: CC BY-NC-ND 4.0; BONES-SEED: its own license). They are not covered by the MIT License.
