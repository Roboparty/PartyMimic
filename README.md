# PartyMimic

**General humanoid motion tracking in minutes on one GPU.** PartyMimic trains one whole-body tracking policy for a
10,731-clip motion library on a **single RTX 4090**:

- **Fast:** 26 minutes of training with the default MLP encoder, 41 with the attention encoder.
- **Precise:** the lowest mean tracking error of every method we compared on five held-out test sets (Loco, Mani, Dance,
  Ground and Extreme martial arts), on two humanoids (Unitree G1 and RoboParty RP1).
- **Sparse references:** it can track from the pelvis and five keypoints (head, wrists, ankles) instead of joint
  angles, and still beats every baseline that gets the full joint reference.

PartyMimic is a model-based learner in the SuperMimic family. It fits an online MLP world model, trains the
policy by backpropagating through 32 imagined steps of that model, and uses a critic beyond the horizon. Its
training recipe is built for generalization and speed. Project page: [roboparty.github.io/PartyMimic](https://roboparty.github.io/PartyMimic/) ([`docs/index.html`](docs/index.html)), with
videos, the recipe ablation and the SuperMimic full-factor ablation.

![Mean held-out error against training compute](docs/static/hero.png)

## GPU memory

Peak memory per card on a 9,173-clip G1 library (300 iterations, `nvidia-smi` once a second). With the default
recipe, one standard 24 GB RTX 4090 is enough (`--ref_enc xattn`: see [Train](#train)):

| Mode | GPU 0 | GPU 1 | it/s |
|---|---:|---:|---:|
| 1 GPU | 22 GB | — | 7.7 |
| 2 GPUs (`--pipeline`) | 14 GB (collection + world model) | 16 GB (policy + critic) | 12.8 |

## Results on held-out motions

Body position error (mm, lower is better) on five held-out test sets, scored by the
[sim2real leaderboard](https://egalahad.github.io/sim2real/leaderboard/) sim2sim evaluator: Loco (80 clips), Mani (48),
Dance (40) and Ground (60) from MotionDecode, a different mocap source, and Extreme (308 KungFuAthlete martial-arts,
jump and flip clips, held out by whole source video). Mean is the mean of the five.

| G1 | Training | Mean | Loco | Mani | Dance | Ground | Extreme |
|---|---:|---:|---:|---:|---:|---:|---:|
| SONIC (official) | 21,000 GPU-h | 58.4 | 28.7 | 45.5 | 48.7 | 80.3 | 88.7 |
| Mimic Lite v1.1 (official) | 50 GPU-h | 67.5 | 25.0 | 28.2 | 43.3 | 145.2 | 95.7 |
| SuperMimic | 52 min | 67.9 | 34.9 | 34.3 | 55.5 | 119.2 | 95.5 |
| NablaMimic | 120 min | 59.9 | 32.9 | 30.3 | 46.9 | 107.8 | 81.6 |
| BlinkMimic | 59 min | 59.9 | 32.9 | 30.9 | 54.1 | 100.1 | 81.4 |
| XMimic | 50 min | 60.2 | 32.3 | 30.6 | 53.5 | 101.3 | 83.1 |
| **PartyMimic**, MLP (default) | 26 min | 45.1 | 25.7 | 20.5 | 36.8 | 80.6 | 62.0 |
| **PartyMimic**, attention | 41 min | 43.5 | 25.1 | 21.1 | 35.0 | 76.2 | 60.3 |

| RP1 | Training | Mean | Loco | Mani | Dance | Ground | Extreme |
|---|---:|---:|---:|---:|---:|---:|---:|
| SuperMimic | 149 min | 74.0 | 34.8 | 29.1 | 53.4 | 162.5 | 90.2 |
| BlinkMimic | 68 min | 60.4 | 33.7 | 27.3 | 50.3 | 110.5 | 80.3 |
| XMimic | 59 min | 62.7 | 33.8 | 29.1 | 49.7 | 121.3 | 79.5 |
| **PartyMimic**, MLP (default) | 26 min | 47.8 | 23.4 | 16.5 | 35.3 | 105.1 | 58.9 |
| **PartyMimic**, attention | 42 min | 44.6 | 22.9 | 16.1 | 33.4 | 94.3 | 56.2 |
| **PartyMimic**, sparse keypoints, MLP | 29 min | 48.7 | 24.7 | 19.1 | 38.3 | 100.9 | 60.7 |
| **PartyMimic**, sparse keypoints, attention | 46 min | 47.8 | 24.1 | 18.2 | 36.8 | 100.0 | 60.1 |

<sub>SuperMimic (paper hyperparameters), NablaMimic (official config), BlinkMimic (main preset), XMimic
(its defaults, BlinkMimic's 12k iterations; npz loader and RP1 scene added) and PartyMimic were all trained by us on the
same 10,731 clips (retargeted per robot), 12k iterations (NablaMimic: its 2-hour budget) and 2 seeds each; the tables give the means. Time is training
wall-clock on one otherwise idle RTX 4090. SONIC and Mimic Lite are official leaderboard entries with different data
and hardware. Completion rates and videos are on the [project page](docs/index.html#results).</sub>

**The PartyMimic recipe**: what matters most, measured by the effect of removing each factor on G1 Loc/Man, the mean of Loco and Mani (ablations on an earlier 9,173-clip library):

| Factor | Effect when removed |
|---|---|
| Muon for policy and world model | Adam instead: +10.7 mm |
| Narrow tracking-reward kernels (`--std_scale 0.35`) | +6.5 mm |
| 32-step imagination | 16 steps: +5.9 mm |
| Reference-action skip | +5.6 mm |
| Policy EMA | +3.4 mm |
| Value-gradient consistency, Muon on all layers, diversity-balanced resets, self-imitation | 0.5–1.6 mm each; Ground +6–19 mm without balanced resets |

The page explains why each factor matters.

## Speed on LAFAN: against XMimic and BlinkMimic (LAFAN-40)

This comparison is on the LAFAN dataset, not the held-out test sets. [XMimic](https://github.com/nico-bohlinger/xmimic)'s benchmark, run on our RTX 4090s with its protocol. All runs
train on its 40 LAFAN1 clips (SHA-256 checked) and are scored on the same clips: 4 replicas, every 100 iterations.
Columns give minutes to BlinkMimic's target, as the mean of 3 seeds. Compilation is excluded for every method by
XMimic's `exclude_compile.py` rule; the start-up column adds it back with warm compilation caches.

| Method | GPUs | it/s | Return ≥ 0.82 | + 0 failed, held ×3 | Held, with start-up | Return at 3,500 it |
|---|---:|---:|---:|---:|---:|---:|
| BlinkMimic (main) | 1 | 7.8 | 2.93 | 3.45 | 3.62 | 0.865 |
| XMimic | 1 | 9.3 | 2.32 | 3.71 | 3.96 | 0.860 |
| **PartyMimic** | 1 | 8.2 | 2.45 | 5.17 | 5.57 | **0.883** |
| XMimic (AL) | 2 | 13.2 | 1.71 | **2.89** | **3.17** | 0.863 |
| **PartyMimic** (`--pipeline`) | 2 | 12.7 | **1.66** | 3.41 | 3.82 | **0.883** |

![LAFAN-40 speed](docs/static/lafan_speed.png)

In the held-out comparison above XMimic matches BlinkMimic (G1 mean error 60.2 vs 59.9 mm). On LAFAN-40,
PartyMimic reaches return 0.82 in the fewest iterations and ends highest. It is slower to zero failures: one or two
hard training clips keep failing for a while. BlinkMimic and XMimic search for those clips with evaluation probes;
PartyMimic dropped the probes because they did not improve held-out tracking. On the second GPU's x1 PCIe link,
the pipeline overlaps all transfers with compute (details on the [page](docs/index.html#speed)).

## Setup

Linux x86_64, an NVIDIA GPU with a CUDA 12.8 driver, and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Roboparty/PartyMimic.git partymimic && cd partymimic
uv sync                    # mjlab, MuJoCo Warp, torch 2.9.0 (CUDA 12.8)
tools/download_data.sh     # G1 description + 40 G1-retargeted LAFAN1 clips -> data/ (enough for a first run)
```

RP1 is not redistributed here. Put its capsule-collision MJCF and meshes (`rp1_28dof_mjlab`) in
`data/rp1_28dof_mjlab/` and its motions in `data/rp1_motions/`, or pass `--motion_dir`.

**Motion data.** `--motion_dir` takes a directory of `.npz` clips, each with `fps` (50), `joint_pos` `[T, J]`,
`body_pos_w` `[T, B, 3]` and `body_quat_w` `[T, B, 4]` (wxyz). The joints and bodies follow the robot's order
(`src/partymimic/<robot>/`), with the pelvis first.
The results above used 10,731 G1 clips: the 40 LAFAN1 clips, a 10,000-clip diversity subset of BONES-SEED, 56
kneeling and crawling clips, and 635 KungFuAthlete clips (ground-corrected, clipped to the joint limits, long clips
split). For RP1, the same motions were retargeted to RP1.

## Train

The defaults of `partymimic.train` are the PartyMimic recipe:

```bash
bash scripts/train.sh logs/g1_s0  --robot g1mj --motion_dir <npz clips> --seed 0   # 12k iterations, ~26 min on a 4090
bash scripts/train.sh logs/rp1_s0 --robot rp1  --motion_dir <npz clips> --seed 0
# two GPUs: collection + world model on cuda:0, policy + critic on cuda:1, overlapped (1.6x the iterations per second)
bash scripts/train.sh logs/g1_p0  --robot g1mj --motion_dir <npz clips> --seed 0 --pipeline
```

With `--pipeline`, the policy learns from the previous iteration's data and world model, and the actor collects
with the policy from the previous update. On the 9,173-clip G1 set this trains 12k iterations in 16 min instead of
30 min, at the same Loc/Man error (mean of Loco and Mani: 23.2 vs 23.6 mm, 2 seeds; Ground 78.3 vs 74.3 mm).

**Options.**

- `--ref_enc xattn`: a cross-attention reference encoder. It is more precise on ground and extreme motions (mean
  error G1 43.5 vs 45.1 mm, RP1 44.6 vs 47.8 mm; tables above) and takes 1.6x the training time on a 4090.
  It needs more than 24 GB. On a 24 GB card, add `--pi_batch 1024`: Ground and Extreme errors are then 2–3 mm
  above the full batch, and Loco, Mani and Dance are unchanged.
- `--interface sparse5`: the policy is given only the pelvis, head, wrist and ankle poses of the reference, with no
  joint angles (e.g. for teleoperation or video pose). A keypoint-to-joint network is fit to the library before
  training and restores the reference-action skip (`--kp_skip`). An auxiliary joint-angle loss is added
  (`--kp_aux 1.0`). Both are on by default with sparse5. On RP1 (2 seeds) its mean error is 48.7 mm with the MLP, against 47.8 mm with joint references (tables above). Export it with `tools/export_deploy_onnx.py` (input
  `ref_kp`).
- `--init <run>/last.pt`: fine-tunes a trained policy on a few clips, e.g. one hard clip before it is deployed. Three
  held-out G1 ground transitions went from 120–415 mm to 24–36 mm in 2k iterations (about 50 s per 250 iterations after
  start-up). It does not help on short, highly dynamic clips.

Checkpoints are written every 2,000 iterations (`ckpt_002000.pt`, ...).
`uv run python -m partymimic.train -h` lists the remaining options.

## Evaluate, export, visualize

```bash
# sim2real leaderboard protocol (ONNX + deploy YAML, MuJoCo CPU sim2sim on MotionDecode)
SIM2REAL=<checkout of github.com/EGalahad/sim2real> bash scripts/eval.sh logs/g1_s0/ckpt_012000.pt eval/g1_s0 g1
# side-by-side videos of the median clip per split: robot solid, reference ghost, both at global poses
MUJOCO_GL=egl python tools/sim2sim_video.py eval/g1_s0 g1 <robot MJCF> videos --sim2real <sim2real checkout>
# other splits / a chosen clip: --splits <split>:<title>[:<clip stem>] ... --note zero-shot
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
