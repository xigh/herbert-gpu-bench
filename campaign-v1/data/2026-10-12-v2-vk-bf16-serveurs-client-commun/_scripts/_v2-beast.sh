#!/usr/bin/env bash
# V2 on the 4080 of beast (clocks locked): the four engines served over HTTP, one after the other,
# timed by the same client (_v2_http.py, Anthropic Messages API, streamed, greedy, one request at a
# time, no prefix cache). herbert-vk-serve 9738be7, vLLM 0.28.0 serve, llama-server 08ffba21e CUDA
# and Vulkan. Each server started in its own process group and stopped with it.
set -u
M=/opt3/models; G=/opt3/bench-qwen35; L=/opt3/llama.cpp-08ffba21e
HS=~/herbert-vk-bf16-serve/target/release/herbert-vk-serve; VL=~/venvs/bench-vllm-cu/bin/vllm
O=~/v2-beast; mkdir -p $O; C=~/_v2_http.py
{ date -u; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; cat ~/herbert-vk-bf16-serve/COMMIT 2>/dev/null; } > $O/env-start.txt 2>&1
wait_ready() { # url seconds
  for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1
}
stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
for m in 0.8B 4B; do
  for e in herbert vllm llama-cuda llama-vk; do
    case $e in
      herbert) setsid $HS $M/Qwen3.5-$m --device nvidia --ctx 4608 --port 8101 --name Qwen3.5-$m > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8101; H=$U/;;
      vllm) VLLM_USE_FLASHINFER_SAMPLER=0 setsid $VL serve $M/Qwen3.5-$m --dtype bfloat16 --max-model-len 4608 --max-num-seqs 8 \
              --gpu-memory-utilization 0.85 --no-enable-prefix-caching --served-model-name Qwen3.5-$m --port 8102 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8102; H=$U/health;;
      llama-cuda) setsid $L/build-cuda/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8103; H=$U/health;;
      llama-vk) setsid $L/build-vk/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev Vulkan1 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8104; H=$U/health;;
    esac
    if wait_ready $H 600; then
      python3 $C --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
      echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
    else
      echo "$m $e: server not ready" ; tail -5 $O/server-$e-$m.log
    fi
    stop $P
  done
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-end.txt
echo ALLDONE-V2-BEAST
