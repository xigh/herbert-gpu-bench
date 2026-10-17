#!/usr/bin/env bash
# Cezanne, 4B, p3 prompt (759 tokens), prefill only (--max-tokens 1, warm): 70e3d5b (wiki #507: 76 tok/s)
# against 2f3ae48 (V1: 64.4), and 2f3ae48 with the two defaults changed since for every vendor turned off
# (conv.cols=0, prefill.pipe=0). 3 rounds, variants interleaved.
set -u
O=~/v1-white2-cezanne/ab; mkdir -p $O
ask=$(cat ~/herbert-vk-bf16-v1/bench/prompts/p3-long.txt)
OLD=~/herbert-vk-bf16-70e3d5b/target/release/herbert-vk-chat; NEW=~/herbert-vk-bf16-v1/target/release/herbert-vk-chat
for r in 1 2 3; do
  $OLD ~/models/Qwen3.5-4B --device amd --no-system --max-tokens 1 --warmup 1 --ask "$ask" > $O/r$r-70e3d5b.log 2>&1
  $NEW ~/models/Qwen3.5-4B --device amd --no-system --max-tokens 1 --warmup 1 --ask "$ask" > $O/r$r-2f3ae48.log 2>&1
  $NEW ~/models/Qwen3.5-4B --device amd --no-system --max-tokens 1 --warmup 1 --tune conv.cols=0 --ask "$ask" > $O/r$r-2f3ae48-convcols0.log 2>&1
  $NEW ~/models/Qwen3.5-4B --device amd --no-system --max-tokens 1 --warmup 1 --tune prefill.pipe=0 --ask "$ask" > $O/r$r-2f3ae48-pipe0.log 2>&1
  echo "ab round $r · $(date -u +%T)"
done
echo ALLDONE-AB
