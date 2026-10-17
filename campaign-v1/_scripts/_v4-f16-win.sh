#!/usr/bin/env bash
# V4 on g5500 (Windows 11, RTX 3060 Ti), run by Git Bash: llama.cpp b11564 in F16 (its strong 16-bit path,
# wiki #544) against the Herbert figures of V1/V2 on this machine. F16 GGUF made here from the BF16 GGUF
# with the official llama-quantize.exe. Bench: llama-bench CUDA and Vulkan at the p3/p4 shapes, N = 5 rounds
# of -r 1. Client: llama-server F16 (CUDA and Vulkan) timed by v2-client, as V2. Clocks every 0.5 s.
# After two crashes of the machine under this load, run under _watchdog-win.sh (90 °C, or the clock log
# frozen), each measurement after a cooldown to 70 °C, 20 s of pause between rounds; ALLDONE-V4-G5500 only when nothing was stopped, else STOPPED-V4-G5500.
set -u
U0=/c/Users/zexig; G=$U0/bench-engines/gguf; LB=$U0/bench-engines/llama-b11564; CL=$U0/_client/target/release/v2-client.exe
O=${O:-$U0/v4-g5500}; mkdir -p $O/bench
{ date -u; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-start.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,fan.speed --format=csv -lms 500 > $O/clocks.csv 2>&1 &
SMI=$!
. "$(dirname "$0")/_watchdog-win.sh"; watchdog $O $$ &
WD=$!
for m in 0.8B 2B; do
  [ -f $G/Qwen3.5-$m-F16.gguf ] || "$LB/cuda/llama-quantize.exe" $G/Qwen3.5-$m-bf16.gguf $G/Qwen3.5-$m-F16.gguf F16 > $O/quantize-$m-F16.log 2>&1
done
sha256sum $G/Qwen3.5-*-F16.gguf > $O/gguf-f16.sha256
for r in 1 2 3 4 5; do
  for m in 0.8B 2B; do
    g=$G/Qwen3.5-$m-F16.gguf
    for N in 759 2880; do
      t=$O/bench/r$r-$m-p$N
      cooldown $O
      "$LB/cuda/llama-bench.exe" -m $g -dev CUDA0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-cuda-f16-pp.json 2> $t-llama-cuda-f16-pp.err
      cooldown $O
      "$LB/cuda/llama-bench.exe" -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-cuda-f16-tg.json 2> $t-llama-cuda-f16-tg.err
      cooldown $O
      "$LB/vulkan/llama-bench.exe" -m $g -dev Vulkan0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-vk-f16-pp.json 2> $t-llama-vk-f16-pp.err
      cooldown $O
      "$LB/vulkan/llama-bench.exe" -m $g -dev Vulkan0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-vk-f16-tg.json 2> $t-llama-vk-f16-tg.err
    done
  done
  echo "bench round $r · $(date -u +%T)"
  [ -f $O/ABORT ] && break
  sleep 20
done
wait_ready() { for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
for m in 0.8B 2B; do
  for e in llama-cuda-f16 llama-vk-f16; do
    [ -f $O/ABORT ] && break 2
    cooldown $O
    case $e in
      llama-cuda-f16) "$LB/cuda/llama-server.exe" -m $G/Qwen3.5-$m-F16.gguf -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103 > $O/server-$e-$m.log 2>&1 & U=http://127.0.0.1:8103;;
      llama-vk-f16) "$LB/vulkan/llama-server.exe" -m $G/Qwen3.5-$m-F16.gguf -dev Vulkan0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & U=http://127.0.0.1:8104;;
    esac
    if wait_ready $U/health 600; then
      "$CL" --prompts $U0/prompts.json --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
      echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
    else
      echo "$m $e: server not ready"; tail -5 $O/server-$e-$m.log
    fi
    taskkill //F //IM llama-server.exe > /dev/null 2>&1; sleep 5
  done
done
kill $SMI 2>/dev/null
date -u > $O/env-end.txt
touch $O/DONE
if [ -f $O/ABORT ]; then echo STOPPED-V4-G5500; else echo ALLDONE-V4-G5500; fi
