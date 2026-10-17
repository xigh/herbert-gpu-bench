#!/usr/bin/env bash
# V4: llama.cpp in F16, its strong 16-bit path (V3, wiki #544: its BF16 prefill is 2.3-3.6 times slower in
# Vulkan), against the Herbert figures of V1/V2 (unchanged: Herbert is bf16 and was measured the same day).
# F16 GGUF made here from the BF16 GGUF with llama-quantize 08ffba21e. Bench: llama-bench at the p3/p4 shapes
# (prefill N, then 255 generated at depth N), N = 5 rounds of -r 1, as bench/vs.sh. Client: llama-server F16
# timed by _v2_http.py, as V2. Settings from the environment:
#   G, L, O        GGUF dir, llama.cpp tree (build-vk, build-cuda), output dir
#   SIZES          "0.8B 4B";  SHAPES "759" or "759 2880"
#   BENCH_VK       Vulkan device pattern for llama-bench ("" = none);  BENCH_CUDA=1 also CUDA0
#   V2_VK          Vulkan device pattern for llama-server ("" = none);  V2_CUDA=1 also CUDA0
#   PROMPTS        the passages of the client (default ~/prompts.json)
set -u
: "${G:?}" "${L:?}" "${O:?}"
SIZES=${SIZES:-0.8B 4B}; SHAPES=${SHAPES:-759}; BENCH_VK=${BENCH_VK:-}; BENCH_CUDA=${BENCH_CUDA:-0}
V2_VK=${V2_VK:-}; V2_CUDA=${V2_CUDA:-0}; PROMPTS=${PROMPTS:-$HOME/prompts.json}
mkdir -p $O
vkdev() { $L/build-vk/bin/llama-bench --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "$1" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/'; }
QB=$L/build-cuda; [ -d $QB ] || QB=$L/build-vk
{ date -u; uname -r; (cd $L && git log -1 --format='%h %ci'); $L/build-vk/bin/llama-bench --list-devices; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-start.txt 2>&1
cmake --build $QB -j $(nproc) --target llama-quantize > $O/build-quantize.log 2>&1
for m in $SIZES; do
  [ -f $G/Qwen3.5-$m-F16.gguf ] || $QB/bin/llama-quantize $G/Qwen3.5-$m-BF16.gguf $G/Qwen3.5-$m-F16.gguf F16 > $O/quantize-$m-F16.log 2>&1
done
sha256sum $G/Qwen3.5-*-F16.gguf > $O/gguf-f16.sha256
if [ -n "$BENCH_VK" ] || [ "$BENCH_CUDA" = 1 ]; then
  D=""; [ -n "$BENCH_VK" ] && D=$(vkdev "$BENCH_VK")
  echo "bench Vulkan device: ${D:-none}" >> $O/env-start.txt
  mkdir -p $O/bench
  for r in 1 2 3 4 5; do
    for m in $SIZES; do
      g=$G/Qwen3.5-$m-F16.gguf
      for N in $SHAPES; do
        t=$O/bench/r$r-$m-p$N
        if [ -n "$D" ]; then
          $L/build-vk/bin/llama-bench -m $g -dev $D -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-vk-f16-pp.json 2> $t-llama-vk-f16-pp.err
          $L/build-vk/bin/llama-bench -m $g -dev $D -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-vk-f16-tg.json 2> $t-llama-vk-f16-tg.err
        fi
        if [ "$BENCH_CUDA" = 1 ]; then
          $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-cuda-f16-pp.json 2> $t-llama-cuda-f16-pp.err
          $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-cuda-f16-tg.json 2> $t-llama-cuda-f16-tg.err
        fi
      done
    done
    echo "bench round $r · $(date -u +%T)"
  done
fi
wait_ready() { for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; kill -TERM $1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL $1 2>/dev/null; sleep 3; }
E=""; [ "$V2_CUDA" = 1 ] && E="llama-cuda-f16"; [ -n "$V2_VK" ] && E="$E llama-vk-f16"
for m in $SIZES; do
  for e in $E; do
    g=$G/Qwen3.5-$m-F16.gguf
    case $e in
      llama-cuda-f16) setsid $L/build-cuda/bin/llama-server -m $g -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8103;;
      llama-vk-f16) setsid $L/build-vk/bin/llama-server -m $g -dev $(vkdev "$V2_VK") -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8104;;
    esac
    if wait_ready $U/health 600; then
      python3 ~/_v2_http.py --prompts $PROMPTS --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
      echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
    else
      echo "$m $e: server not ready"; tail -5 $O/server-$e-$m.log
    fi
    stop $P
  done
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-end.txt
echo ALLDONE-V4
