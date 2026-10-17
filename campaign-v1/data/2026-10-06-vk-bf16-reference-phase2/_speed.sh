#!/usr/bin/env bash
# speed.sh MODEL OUTDIR ROUNDS DEVICE... — p3-long (759 prompt tokens, 255
# generated, thinking on, greedy) on the CPU oracle and on each Vulkan device
# and variant, interleaved round after round; full outputs in
# OUTDIR/r<round>-<engine>.log. VARIANTS as in compare.sh; NOCPU=1 skips the CPU.
set -u
MODEL=$1; OUT=$2; ROUNDS=$3; shift 3
mkdir -p "$OUT"
cd "$(dirname "$0")"
ask=$(cat prompts/p3-long.txt)
VARIANTS=${VARIANTS:-default:}
IFS=';' read -ra vs <<< "$VARIANTS"
for r in $(seq 1 "$ROUNDS"); do
  echo "round $r · $(uptime)"
  if [ "${NOCPU:-0}" != 1 ]; then
    ~/herbert-jit-bf16/target/release/herbert-bf16-chat "$MODEL" --no-system --max-tokens 256 --ask "$ask" > "$OUT/r$r-cpu.log" 2>&1
  fi
  for v in "${vs[@]}"; do
    vname=${v%%:*}; vargs=${v#*:}
    for d in "$@"; do
      # shellcheck disable=SC2086
      ../target/release/herbert-vk-chat "$MODEL" --device "$d" --no-system --max-tokens 256 $vargs --ask "$ask" > "$OUT/r$r-vk-$d-$vname.log" 2>&1
    done
  done
done
echo done
