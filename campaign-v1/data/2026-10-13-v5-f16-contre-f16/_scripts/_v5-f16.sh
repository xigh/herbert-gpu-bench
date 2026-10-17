#!/usr/bin/env bash
# V5: f16 against f16. Herbert in f16 (herbert-vk-f16, default build: weights, activations, cache and cooperative
# matrices in IEEE f16; on AMD without matrices, the SIMT GEMM in packed f16, gemm.h16=1) against llama.cpp
# 08ffba21e in F16, its strong 16-bit path (V3, wiki #544). Herbert bf16 from the same tree
# (--no-default-features --features bf16: herbert-vk-bf16's engine bit for bit) in the same rounds, so that the
# two same-format comparisons (bf16/bf16 from V1, f16/f16 here) sit next to each other. V4 had put Herbert bf16
# against llama.cpp F16: two formats, not a comparison.
# Bench: p3/p4 shapes (prefill N, then 255 generated at depth N), N = 5 rounds, the engines interleaved inside
# each round (Herbert warm, llama-bench -r 1). Client: herbert-vk-serve bf16 and f16, llama-server F16, timed by
# _v2_http.py as V2. Settings from the environment:
#   M, G, L, F, O  models dir, GGUF dir, llama.cpp tree (build-vk, build-cuda), herbert-vk-f16 tree, output dir
#   HDEV           Herbert's --device;  VKPAT  llama.cpp Vulkan device pattern;  CUDA=1 also llama.cpp CUDA0
#   SIZES "0.8B 4B";  SHAPES "759" or "759 2880";  CLIENT=1 the client pass;  PROMPTS (default ~/prompts.json)
set -u
: "${M:?}" "${G:?}" "${L:?}" "${F:?}" "${O:?}" "${HDEV:?}" "${VKPAT:?}"
CUDA=${CUDA:-0}; SIZES=${SIZES:-0.8B 4B}; SHAPES=${SHAPES:-759}; CLIENT=${CLIENT:-0}; PROMPTS=${PROMPTS:-$HOME/prompts.json}
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
mkdir -p $O/bench
LDEV=$($L/build-vk/bin/llama-bench --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "$VKPAT" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
[ -n "$LDEV" ] || { echo "no llama.cpp device for '$VKPAT'"; exit 1; }
( cd $F && cargo build --release --locked --features serve ) > $O/build-f16.log 2>&1 || { echo "build f16 failed"; exit 1; }
( cd $F && cargo build --release --locked --no-default-features --features "bf16 serve" --target-dir target-bf16 ) > $O/build-bf16.log 2>&1 || { echo "build bf16 failed"; exit 1; }
HB=$F/target-bf16/release; HF=$F/target/release
{ date -u; uname -r; cat $F/COMMIT; (cd $L && git log -1 --format='%h %ci'); $L/build-vk/bin/llama-bench --list-devices; echo "llama.cpp Vulkan device: $LDEV"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv
  sha256sum $HB/herbert-vk-chat $HF/herbert-vk-chat $HB/herbert-vk-serve $HF/herbert-vk-serve $G/Qwen3.5-*-F16.gguf; } > $O/env-start.txt 2>&1
SMI=""; if [ "$CUDA" = 1 ]; then nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -lms 500 > $O/clocks.csv 2>&1 & SMI=$!; fi
for r in 1 2 3 4 5; do
  for m in $SIZES; do
    g=$G/Qwen3.5-$m-F16.gguf
    for N in $SHAPES; do
      t=$O/bench/r$r-$m-p$N
      case $N in 759) ask=$(cat $F/bench/prompts/p3-long.txt);; 2880) ask=$(cat $F/bench/prompts/p4-xlong.txt);; esac
      $HB/herbert-vk-chat $M/Qwen3.5-$m --device $HDEV --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $t-herbert-bf16.log 2>&1
      $HF/herbert-vk-chat $M/Qwen3.5-$m --device $HDEV --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $t-herbert-f16.log 2>&1
      $L/build-vk/bin/llama-bench -m $g -dev $LDEV -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-vk-f16-pp.json 2> $t-llama-vk-f16-pp.err
      $L/build-vk/bin/llama-bench -m $g -dev $LDEV -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-vk-f16-tg.json 2> $t-llama-vk-f16-tg.err
      if [ "$CUDA" = 1 ]; then
        $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-cuda-f16-pp.json 2> $t-llama-cuda-f16-pp.err
        $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-cuda-f16-tg.json 2> $t-llama-cuda-f16-tg.err
      fi
    done
  done
  echo "bench round $r · $(date -u +%T)"
done
if [ "$CLIENT" = 1 ]; then
  wait_ready() { for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
  stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
  E="herbert-bf16 herbert-f16 llama-vk-f16"; [ "$CUDA" = 1 ] && E="$E llama-cuda-f16"
  for m in $SIZES; do
    g=$G/Qwen3.5-$m-F16.gguf
    for e in $E; do
      case $e in
        herbert-bf16) setsid $HB/herbert-vk-serve $M/Qwen3.5-$m --device $HDEV --ctx 4608 --port 8101 --name Qwen3.5-$m > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8101; H=$U/;;
        herbert-f16) setsid $HF/herbert-vk-serve $M/Qwen3.5-$m --device $HDEV --ctx 4608 --port 8102 --name Qwen3.5-$m > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8102; H=$U/;;
        llama-vk-f16) setsid $L/build-vk/bin/llama-server -m $g -dev $LDEV -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8104; H=$U/health;;
        llama-cuda-f16) setsid $L/build-cuda/bin/llama-server -m $g -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8103; H=$U/health;;
      esac
      if wait_ready $H 600; then
        python3 ~/_v2_http.py --prompts $PROMPTS --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
        echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
      else
        echo "$m $e: server not ready"; tail -5 $O/server-$e-$m.log
      fi
      stop $P
    done
  done
fi
[ -n "$SMI" ] && kill $SMI 2>/dev/null
{ date -u; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-end.txt 2>&1
echo ALLDONE-V5
