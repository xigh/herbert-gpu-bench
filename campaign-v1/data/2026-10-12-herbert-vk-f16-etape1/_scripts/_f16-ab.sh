#!/usr/bin/env bash
# f16 against bf16, the same tree (herbert-vk-f16: default build = f16, --no-default-features --features bf16
# = herbert-vk-bf16 bit for bit): p3 shape (759 + 255), warm, variants interleaved, N rounds; then the
# tokens of p1-p4 against the CPU oracle with the 5 best logits. Usage: _f16-ab.sh OUT DEVICE ROUNDS [SIZES]
set -u
O=$1; DEV=$2; N=$3; SIZES=${4:-0.8B 4B}
MODELS=${MODELS:-$HOME/models}; ORACLE=${ORACLE:-$HOME/herbert-jit-bf16/target/release/herbert-bf16-chat}; EXE=${EXE:-}
cd "$(dirname "$0")/.."; mkdir -p $O
ask=$(cat bench/prompts/p3-long.txt)
for r in $(seq 1 $N); do
  for m in $SIZES; do
    for v in f16:target bf16:target-bf16; do
      n=${v%%:*}; d=${v#*:}
      $d/release/herbert-vk-chat$EXE $MODELS/Qwen3.5-$m --device $DEV --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $O/r$r-$m-$n.log 2>&1
    done
  done
  echo "round $r · $(date -u +%T)"
done
for m in $SIZES; do
  mkdir -p $O/cmp-$m
  for p in bench/prompts/p*.txt; do
    name=$(basename $p .txt); think="--no-think"; max=200
    case $name in p3-*) think=""; max=256;; esac
    [ -f $O/cmp-$m/$name-cpu.log ] || "$ORACLE" $MODELS/Qwen3.5-$m --no-system $think --max-tokens $max --ask "$(cat $p)" > $O/cmp-$m/$name-cpu.log 2>&1
    for v in f16:target bf16:target-bf16; do
      n=${v%%:*}; d=${v#*:}
      HERBERT_VK_TOP=5 $d/release/herbert-vk-chat$EXE $MODELS/Qwen3.5-$m --device $DEV --no-system $think --max-tokens $max --ask "$(cat $p)" > $O/cmp-$m/$name-vk-$n-top.log 2>&1
    done
  done
  echo "tokens $m · $(date -u +%T)"
done
echo ALLDONE-AB
