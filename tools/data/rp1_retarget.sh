#!/bin/bash
# G1 -> RP1 retarget with human-humanoid-tools' robot-to-robot IK (R2R), as for the RP1 library and test sets:
#   HHT=<human-humanoid-tools checkout> tools/data/rp1_retarget.sh <G1 CSV dir> <out dir> [source fps, default 50]
# Input: tools/data/rp1_data.py g1csv (50 Hz) or md_in (120 Hz) output; its shard_* subdirectories are retargeted one
# after another into <out dir>/shard_* (run several copies on separate input dirs to use more GPUs). Robot presets:
# assets/hhtools_robots (G1 source, RP1 target with the zero-pose calibration), meshes from tools/download_data.sh.
# Resumable: a rerun skips finished clips (a Newton crash kills the worker, so each shard is retried).
set -u
IN=$(realpath "${1:?G1 CSV dir}"); OUT=$(realpath -m "${2:?out dir}"); SFPS=${3:-50}
HHT=$(realpath "${HHT:?set HHT to a checkout of github.com/Roboparty/human-humanoid-tools}")
cd "$(dirname "$0")/../.."
PY=${HHT_PYTHON:-$HHT/.venv/bin/python}
R=$PWD/data/hhtools_robots
for r in g1 rp1; do mkdir -p "$R/$r" && cp assets/hhtools_robots/$r/* "$R/$r/"; done
ln -sfn "$PWD/data/g1_description/meshes" "$R/g1/meshes"
ln -sfn "$PWD/data/rp1_28dof_mjlab/meshes" "$R/rp1/meshes"
export HHTOOLS_ROBOT_PATH=$R PYTHONUNBUFFERED=1
shards=$(cd "$IN" && ls -d shard_* 2>/dev/null) || shards=.
for s in $shards; do
  mkdir -p "$OUT/$s"
  for try in 1 2 3 4 5; do
    (cd "$HHT" && "$PY" scripts/batch_r2r_retarget.py --source-robot g1 --target-robot rp1 --in "$IN/$s" --out "$OUT/$s" \
      --profile mimic --backend newton --source-fps "$SFPS" --fps 50 --ik-iterations 24 --skip-existing --in-process \
      --failure-log "$OUT/$s/failures.jsonl" >> "$OUT/$s/log.txt" 2>&1) && break
    echo "try $try exit $?" >> "$OUT/$s/log.txt"
  done
  echo "$s: $(find "$OUT/$s" -name '*.csv' -size +0 | wc -l) clips"
done
