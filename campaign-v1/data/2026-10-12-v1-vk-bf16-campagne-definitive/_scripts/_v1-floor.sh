#!/usr/bin/env bash
# V1 complement on the 4080 of beast (clocks locked): Herbert's own latency with short prompts, the
# counterpart of vLLM's at 1, 16 and 64 tokens. Prompts cut from pp128 word by word until Herbert counts
# exactly 16 and 64 prompt tokens (template included, as for pp128); then N = 5, warm, one token generated.
set -u
H=~/herbert-vk-bf16-v1; HV=$H/target/release/herbert-vk-chat; M=/opt3/models; O=~/v1-beast/floor; mkdir -p $O
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-start.txt
W=($(cat $H/bench/prompts/pp128.txt))  # every word, across lines
count() { $HV $M/Qwen3.5-0.8B --device nvidia --no-system --no-think --max-tokens 1 --ask "$1" 2>&1 | grep -o -E '^\[turn 1\] [0-9]+' | awk '{print $3}'; }
for target in 16 64; do
  for n in $(seq 1 ${#W[@]}); do
    text="${W[*]:0:$n}"; c=$(count "$text")
    if [ "$c" = "$target" ]; then echo -n "$text" > $O/pp$target.txt; echo "pp$target: $n words, $c tokens"; break; fi
    [ "$c" -gt "$target" ] && { echo "pp$target: overshoot at $n words ($c)"; break; }
  done
done
for r in 1 2 3 4 5; do
  for m in 0.8B 4B; do
    for p in 16 64; do
      [ -f $O/pp$p.txt ] && $HV $M/Qwen3.5-$m --device nvidia --no-system --no-think --max-tokens 1 --warmup 1 --ask "$(cat $O/pp$p.txt)" > $O/r$r-$m-herbert-pp$p.log 2>&1
    done
  done
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-end.txt
for f in $O/r*-herbert-pp*.log; do printf "%s " $(basename $f); grep -h '^\[turn' $f; done
echo ALLDONE-FLOOR
