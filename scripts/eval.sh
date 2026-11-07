#!/bin/bash
# Score a checkpoint with the sim2real leaderboard protocol: export to ONNX + deploy YAML, then MuJoCo (CPU) sim2sim
# on the MotionDecode splits (locomotion, manipulation, ground, dance), every clip from its first frame, seed 0.
# Usage: SIM2REAL=<checkout of github.com/EGalahad/sim2real> bash scripts/eval.sh <ckpt.pt> <out_dir> [g1|rp1]
# rp1 reads the RP1-retargeted MotionDecode clips from $SIM2REAL/datasets/motiondecode_rp1.
# Per-split summaries land in <out_dir>/<split>/; trajectories (for tools/sim2sim_video.py) next to them.
set -eu
cd "$(dirname "$0")/.."
CKPT=${1:?usage: bash scripts/eval.sh <ckpt.pt> <out_dir> [g1|rp1]}; OUT=$(realpath -m "${2:?out_dir}"); ROBOT=${3:-g1}
S2R=${SIM2REAL:?set SIM2REAL to a checkout of github.com/EGalahad/sim2real}
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ${PYTHON:-uv run python} tools/export_deploy_onnx.py "$CKPT" "$OUT/policy.onnx"
DSET=(); [ "$ROBOT" = g1 ] || DSET=(--datasets-root "datasets/motiondecode_$ROBOT")
cd "$S2R"
SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/tracking_experiment/run_motiondecode_eval.py \
  --splits locomotion manipulation ground dance --policy "partymimic=$OUT/policy.yaml" --output-dir "$OUT" \
  --max-workers "${WORKERS:-8}" --robot "$ROBOT" "${DSET[@]}"
