#!/usr/bin/env bash
# V1 on beast (protocol.md): the 4080 (SM 2 805 MHz, memory 11 201 MHz locked) and the
# Raphael iGPU. herbert-vk-bf16 2f3ae48 (default build = bf16), llama.cpp 08ffba21e (b11564) CUDA and
# Vulkan, vLLM 0.28.0. Phases: env, T (tokens against the CPU oracle with the 5 best logits),
# P (prefill against the length, vLLM's fixed cost), D3/D4 (p3 and p4 shapes), Q (llama-bench settings),
# R (Raphael iGPU), env again. Every output kept in full under $O.
set -u
H=~/herbert-vk-bf16-v1; HV=$H/target/release/herbert-vk-chat
L=/opt3/llama.cpp-08ffba21e; LC=$L/build-cuda/bin/llama-bench; LV=$L/build-vk/bin/llama-bench
G=/opt3/bench-qwen35; M=/opt3/models; VL=~/venvs/bench-vllm-cu/bin/vllm
O=~/v1-beast; mkdir -p $O
envs() {
  { date -u; uptime; cat /proc/driver/nvidia/version; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,persistence_mode --format=csv
    cat $H/COMMIT; (cd $L && git log -1 --format='%h %ci' && git rev-list --count HEAD); $LV --list-devices; $LC --list-devices
    ~/venvs/bench-vllm-cu/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"; } > $O/env-$1.txt 2>&1
}
envs start
cp ~/herbert-vk-bf16/bench/prompts/pp*.txt $H/bench/prompts/
mkdir -p $O/prompts; cp $H/bench/prompts/*.txt $O/prompts/

echo "T · $(date -u +%T)"
for m in 0.8B 4B; do
  mkdir -p $O/cmp-$m; cp ~/herbert-vk-bf16/runs/dr2/cmp-$m/*-cpu.log $O/cmp-$m/
  for p in $H/bench/prompts/*.txt; do
    name=$(basename "$p" .txt); think="--no-think"; max=200; ctx=""
    case $name in p3-*) think=""; max=256;; pp4000) ctx="--ctx 4608";; esac
    HERBERT_VK_TOP=5 $HV $M/Qwen3.5-$m --device nvidia --no-system $think --max-tokens $max $ctx --ask "$(cat "$p")" > $O/cmp-$m/$name-vk-nvidia-top.log 2>&1
    echo "T $m $name exit $?"
  done
done

echo "P · $(date -u +%T)"
PP="128 256 512 1024 2048 4000"
for r in 1 2 3 4 5; do
  for m in 0.8B 4B; do
    for p in $PP; do
      $HV $M/Qwen3.5-$m --device nvidia --no-system --no-think --max-tokens 1 --warmup 1 --ask "$(cat $O/prompts/pp$p.txt)" > $O/pp-r$r-$m-herbert-pp$p.log 2>&1
    done
  done
  echo "P herbert round $r"
done
for m in 0.8B 4B; do
  g=$G/Qwen3.5-$m-BF16.gguf
  $LC -m $g -dev CUDA0 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/pp-$m-llama-cuda.json 2> $O/pp-$m-llama-cuda.err
  $LV -m $g -dev Vulkan1 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/pp-$m-llama-vk.json 2> $O/pp-$m-llama-vk.err
  echo "P llama $m"
done
for m in 0.8B 4B; do
  for p in 1 16 64 $PP; do
    VLLM_USE_FLASHINFER_SAMPLER=0 $VL bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len $p --output-len 1 \
      --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
      --output-json $O/pp-vllm-$m-pp$p.json > $O/pp-vllm-$m-pp$p.log 2>&1
    echo "P vllm $m pp$p exit $?"
  done
done

echo "D3 · $(date -u +%T)"
( cd $H && LLCUDA=$LC LLVK=$LV GGUF=$G MODELS=$M HVBIN=$HV PAIRS="Vulkan1:nvidia:nv" CUDAPAIR="nvidia:nv" WARM=1 \
  TUNE="--warmup 1 --tune host.poll=1" VLLM=1 bench/vs.sh $O/p3 5 ) > $O/p3.out 2>&1

echo "D4 · $(date -u +%T)"
ask=$(cat $O/prompts/p4-xlong.txt); mkdir -p $O/p4
for r in 1 2 3 4 5; do
  for m in 0.8B 4B; do
    g=$G/Qwen3.5-$m-BF16.gguf
    $LC -m $g -dev CUDA0 -ngl 99 -p 2880 -n 0 -r 1 -o json > $O/p4/r$r-$m-llama-cuda-pp.json 2> $O/p4/r$r-$m-llama-cuda-pp.err
    $LC -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d 2880 -r 1 -o json > $O/p4/r$r-$m-llama-cuda-tg.json 2> $O/p4/r$r-$m-llama-cuda-tg.err
    $LV -m $g -dev Vulkan1 -ngl 99 -p 2880 -n 0 -r 1 -o json > $O/p4/r$r-$m-llama-vk-pp.json 2> $O/p4/r$r-$m-llama-vk-pp.err
    $LV -m $g -dev Vulkan1 -ngl 99 -p 0 -n 255 -d 2880 -r 1 -o json > $O/p4/r$r-$m-llama-vk-tg.json 2> $O/p4/r$r-$m-llama-vk-tg.err
    $HV $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $O/p4/r$r-$m-herbert.log 2>&1
    $HV $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 256 --warmup 1 --tune host.poll=1 --ask "$ask" > $O/p4/r$r-$m-herbert-poll.log 2>&1
  done
  echo "D4 round $r"
done
for m in 0.8B 4B; do
  for len in 255 1; do
    VLLM_USE_FLASHINFER_SAMPLER=0 $VL bench latency --model $M/Qwen3.5-$m --dtype bfloat16 --input-len 2880 --output-len $len \
      --batch-size 1 --num-iters-warmup 2 --num-iters 5 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
      --output-json $O/p4/vllm-$m-out$len.json > $O/p4/vllm-$m-out$len.log 2>&1
    echo "D4 vllm $m out$len exit $?"
  done
done

echo "Q · $(date -u +%T)"
for m in 0.8B 4B; do
  g=$G/Qwen3.5-$m-BF16.gguf
  $LC -m $g -dev CUDA0 -ngl 99 -p 128,512,2048,4000 -n 0 -ub 256,512,1024,2048 -fa 0,1 -r 3 -o json > $O/q-$m-llama-cuda.json 2> $O/q-$m-llama-cuda.err
  $LV -m $g -dev Vulkan1 -ngl 99 -p 128,512,2048,4000 -n 0 -ub 256,512,1024,2048 -fa 0,1 -r 3 -o json > $O/q-$m-llama-vk.json 2> $O/q-$m-llama-vk.err
  echo "Q $m"
done
envs mid

echo "R · $(date -u +%T)"
( cd $H && LLCUDA="" LLVK=$LV GGUF=$G MODELS=$M HVBIN=$HV PAIRS="Vulkan0:amd:raphael" CUDAPAIR="" WARM=1 \
  bench/vs.sh $O/p3-raphael 5 ) > $O/p3-raphael.out 2>&1
envs end
echo ALLDONE-V1-BEAST
