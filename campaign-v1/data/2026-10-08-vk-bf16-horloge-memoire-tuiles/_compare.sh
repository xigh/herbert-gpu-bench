#!/usr/bin/env bash
# compare.sh MODEL OUTDIR DEVICE... — runs every prompt of prompts/ on the CPU
# oracle (herbert-bf16-chat) and on each Vulkan device (herbert-vk-chat),
# greedy, no system prompt; the full outputs go to OUTDIR/<prompt>-<engine>.log.
# p3 runs with thinking on (the template's <think> block), the others without.
# VARIANTS="name:--tune a=1 --tune b=2;other:..." runs each Vulkan device once
# per variant (default: one variant "default" with no option); the CPU runs once.
# A variant's options starting with @BINARY run that build instead.
set -u
MODEL=$1; OUT=$2; shift 2
mkdir -p "$OUT"
cd "$(dirname "$0")"
VARIANTS=${VARIANTS:-default:}
for p in prompts/*.txt; do
  name=$(basename "$p" .txt)
  ask=$(cat "$p")
  think="--no-think"; max=200
  case $name in p3-*) think=""; max=256;; esac
  if [ ! -f "$OUT/$name-cpu.log" ]; then
    ~/herbert-jit-bf16/target/release/herbert-bf16-chat "$MODEL" --no-system $think --max-tokens $max --ask "$ask" > "$OUT/$name-cpu.log" 2>&1
    echo "$name cpu exit $?"
  fi
  IFS=';' read -ra vs <<< "$VARIANTS"
  for v in "${vs[@]}"; do
    vname=${v%%:*}; vargs=${v#*:}
    for d in "$@"; do
      # shellcheck disable=SC2086
      bin=../target/release/herbert-vk-chat
      # A variant "name:@BINARY options" runs another build (e.g. --features bf16).
      if [[ $vargs == @* ]]; then bin=${vargs%% *}; bin=${bin#@}; vargs=${vargs#* }; [[ $vargs == @* ]] && vargs=""; fi
      "$bin" "$MODEL" --device "$d" --no-system $think --max-tokens $max $vargs --ask "$ask" > "$OUT/$name-vk-$d-$vname.log" 2>&1
      echo "$name vk-$d-$vname exit $?"
    done
  done
done
