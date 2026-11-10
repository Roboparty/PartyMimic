#!/bin/bash
# Training libraries and test sets of the results, from the public sources (README, "Data"):
#   tools/data/build_library.sh g1          G1 library data/motions/g1_mix (LAFAN1 + BONES-SEED subset + KungfuAthlete
#                                           train split) and the Extreme test set data/testsets/extreme_g1; its steps
#                                           alone: lafan, bones, kungfu (also the Extreme set), mix
#   HHT=<human-humanoid-tools> tools/data/build_library.sh rp1
#                                           after g1: G1 -> RP1 R2R of both, -> data/motions/rp1_mix, data/testsets/extreme_rp1
#   HHT=... SIM2REAL=<sim2real> tools/data/build_library.sh rp1_md
#                                           RP1 copy of the MotionDecode held-out splits -> $SIM2REAL/datasets/motiondecode_rp1
# Needs tools/download_data.sh first. BONES-SEED is gated on Hugging Face: accept its terms, then `hf auth login` or
# set HF_TOKEN. Disk: the 23.5 GB BONES archive (delete it after the bones step), 13 GB of G1 npz; RP1: 5 GB of csv,
# 1 GB of npz.
# R2R takes ~8 s per clip per process: P=<processes> GPUS="0 1" runs P processes over those GPUs.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-uv run python}
HFD=(uvx --from huggingface_hub hf download --repo-type dataset)
T=tools/data; RAW=data/raw; M=data/motions; TS=data/testsets
P=${P:-4}; GPUS=(${GPUS:-0})
log() { echo "[$(date '+%F %T')] $*"; }

retarget() {  # <G1 CSV dir with shard_*> <out> <source fps>
  local i=0 s
  for s in "$1"/shard_*; do
    CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} $T/rp1_retarget.sh "$s" "$2/$(basename "$s")" "$3" &
    i=$((i + 1))
  done
  wait
}

lafan() {
  log "LAFAN1"
  $PY $T/prep_motions.py --format lafan --src data/lafan1_g1 --out $M/g1/lafan
}

bones() {
  log "BONES-SEED (download 23.5 GB, extract the 10,000 clips of manifests/bones_seed_div10k.txt)"
  "${HFD[@]}" bones-studio/seed g1.tar.gz --local-dir $RAW/bones
  tar -xzf $RAW/bones/g1.tar.gz -C $RAW/bones -T manifests/bones_seed_div10k.txt
  $PY $T/prep_motions.py --format bones --src $RAW/bones --list manifests/bones_seed_div10k.txt --out $M/g1/bones
}

kungfu() {
  log "KungfuAthlete"
  "${HFD[@]}" LuluCao/KungfuAthleteBot --local-dir $RAW/kungfu \
    --include "kungfu_athlete_for_g1_29dof/datasets/org_smooth/*.npz" --include "Action File Title.csv"
  local K=$RAW/kungfu/kungfu_athlete_for_g1_29dof/datasets/org_smooth
  $PY $T/kungfu.py qa $K "$RAW/kungfu/Action File Title.csv" $RAW/kungfu/qa.json
  $PY $T/kungfu.py write $RAW/kungfu/qa.json manifests/kungfu_split.json $RAW/kungfu/clean
  $PY $T/prep_motions.py --src $RAW/kungfu/clean/train_q50 --fps-in 50 --out $M/g1/kungfu
  mkdir -p $TS && rm -rf $TS/extreme_g1 && mv $RAW/kungfu/clean/test_any4hdmi $TS/extreme_g1
  log "Extreme: $(wc -l < $TS/extreme_g1/motion-list.txt) clips"
}

mix() {
  rm -rf $M/g1_mix && mkdir -p $M/g1_mix
  for d in lafan bones kungfu; do
    if [ -d $M/g1/$d ]; then ln -s "$PWD/$M/g1/$d"/*.npz $M/g1_mix/; fi
  done
  log "G1 library $M/g1_mix: $(ls $M/g1_mix | wc -l) clips"
}

g1() { lafan; bones; kungfu; mix; }

rp1() {
  : "${HHT:?set HHT to a human-humanoid-tools checkout}"
  local R=$RAW/rp1
  log "RP1 library: G1 CSVs -> R2R -> npz"
  $PY $T/rp1_data.py g1csv $M/g1_mix $R/g1csv --shards "$P"
  retarget $R/g1csv $R/rp1csv 50
  $PY $T/rp1_data.py npz $R/rp1csv $M/rp1_mix
  $PY $T/rp1_floor.py $R/floor_library.json $M/rp1_mix

  log "RP1 Extreme set"
  $PY $T/prep_motions.py --src $TS/extreme_g1/motions --fps-in 50 --out $R/extreme_npz
  $PY $T/rp1_data.py g1csv $R/extreme_npz $R/extreme_g1csv --shards "$P"
  retarget $R/extreme_g1csv $R/extreme_rp1csv 50
  $PY $T/rp1_data.py hdxset $R/extreme_rp1csv $TS/extreme_g1 $TS/extreme_rp1
  $PY $T/rp1_floor.py $R/floor_extreme.json $TS/extreme_rp1
  log "RP1 library $M/rp1_mix: $(ls $M/rp1_mix | wc -l) clips; Extreme: $(wc -l < $TS/extreme_rp1/motion-list.txt) clips"
}

rp1_md() {
  : "${HHT:?set HHT}" "${SIM2REAL:?set SIM2REAL to a sim2real checkout with datasets/motiondecode}"
  local R=$RAW/rp1 MD=$SIM2REAL/datasets/motiondecode
  log "MotionDecode source CSVs of the 228 held-out clips (CMRobot/MotionDecode)"
  mapfile -t inc < <(python3 -c "import csv,sys; [print('samples/' + r['path']) for s in ('locomotion','manipulation',
    'ground','dance') for r in csv.DictReader(open(f'$MD/{s}/selection_manifest.csv'))]")
  "${HFD[@]}" CMRobot/MotionDecode "${inc[@]}" --local-dir $RAW/motiondecode
  $PY $T/rp1_data.py md_in $R/md_g1csv --shards "$P" --md $MD --md-src $RAW/motiondecode/samples
  retarget $R/md_g1csv $R/md_rp1csv 120
  $PY $T/rp1_data.py testset $R/md_rp1csv "$SIM2REAL/datasets/motiondecode_rp1" --md $MD
}

[ $# -gt 0 ] || { sed -n '2,12p' "$0"; exit 1; }
for stage in "$@"; do "$stage"; done
