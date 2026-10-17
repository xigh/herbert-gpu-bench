#!/usr/bin/env bash
# pp3: prefill against the prompt length on the 4080 of beast (2 805 MHz SM and 11 201 MHz memory locked):
# herbert (bf16 build, NVIDIA defaults at 5922789, warm, N = 5 rounds), llama.cpp c841aeeb8
# CUDA and Vulkan (llama-bench -p list, -r 5), vLLM (its latency bench, out 1, 5 iterations).
cd ~/herbert-vk-bf16
O=runs/pp3; mkdir -p $O
/opt3/llama.cpp/build-vk/bin/llama-bench --list-devices > $O/devices.txt 2>&1
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-start.txt
PP="128 256 512 1024 2048 4000"
for r in 1 2 3 4 5; do
  echo "herbert round $r · $(uptime)"
  for m in 0.8B 4B; do
    for p in $PP; do
      target-bf16/release/herbert-vk-chat /opt3/models/Qwen3.5-$m --device nvidia --no-system --no-think --max-tokens 1 --warmup 1 --ask "$(cat $O/prompts/pp$p.txt)" > $O/r$r-$m-herbert-pp$p.log 2>&1
    done
  done
done
for m in 0.8B 4B; do
  g=/opt3/bench-qwen35/Qwen3.5-$m-BF16.gguf
  /opt3/llama.cpp/build/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/$m-llama-cuda.json 2> $O/$m-llama-cuda.err
  /opt3/llama.cpp/build-vk/bin/llama-bench -m $g -dev Vulkan0 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/$m-llama-vk.json 2> $O/$m-llama-vk.err
  echo "llama $m done · $(uptime)"
done
for m in 0.8B 4B; do
  for p in $PP; do
    VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/bench-vllm-cu/bin/vllm bench latency --model /opt3/models/Qwen3.5-$m --dtype bfloat16 --input-len $p --output-len 1 \
      --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
      --output-json $O/vllm-$m-pp$p.json > $O/vllm-$m-pp$p.log 2>&1
    echo "vllm $m pp$p exit $?"
  done
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-end.txt
echo ALLDONE-PP3
