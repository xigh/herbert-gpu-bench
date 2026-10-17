#!/usr/bin/env bash
# Step 3: the SIMT GEMM in packed f16 (gemm.h16 = the flush period, in tiles of 16) against the f32 one,
# the same f16 build, interleaved, p3 shape warm, N rounds; then the tokens of p1-p4 against the CPU oracle
# (CPU logs of an earlier _f16-ab.sh run reused) with the 5 best logits. Usage: _h16-ab.sh OUT DEVICE N CPU_DIR [SIZES]
set -u
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
O=$1; DEV=$2; N=$3; CPU=$4; SIZES=${5:-0.8B 4B}
MODELS=${MODELS:-$HOME/models}
cd "$(dirname "$0")/.."; mkdir -p $O
cargo build --release --locked > $O/build.log 2>&1 || { echo "build failed"; exit 1; }
V="f32: h1:gemm.h16=1 h2:gemm.h16=2 h4:gemm.h16=4 h64:gemm.h16=64"
ask=$(cat bench/prompts/p3-long.txt)
for r in $(seq 1 $N); do
  for m in $SIZES; do
    for v in $V; do
      n=${v%%:*}; t=${v#*:}
      target/release/herbert-vk-chat $MODELS/Qwen3.5-$m --device $DEV --no-system --max-tokens 256 --warmup 1 ${t:+--tune $t} --ask "$ask" > $O/r$r-$m-$n.log 2>&1
    done
  done
  echo "round $r · $(date -u +%T)"
done
for m in $SIZES; do
  mkdir -p $O/cmp-$m; cp $CPU/cmp-$m/*-cpu.log $O/cmp-$m/
  for p in bench/prompts/p*.txt; do
    name=$(basename $p .txt); think="--no-think"; max=200
    case $name in p3-*) think=""; max=256;; esac
    for v in $V; do
      n=${v%%:*}; t=${v#*:}
      HERBERT_VK_TOP=5 target/release/herbert-vk-chat $MODELS/Qwen3.5-$m --device $DEV --no-system $think --max-tokens $max ${t:+--tune $t} --ask "$(cat $p)" > $O/cmp-$m/$name-vk-$n-top.log 2>&1
    done
  done
  echo "tokens $m · $(date -u +%T)"
done
echo ALLDONE-H16
