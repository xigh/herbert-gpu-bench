#!/usr/bin/env bash
# V3 on the 4080 of beast (clocks locked): is bf16 llama.cpp's weak path? llama-bench 08ffba21e, CUDA and
# Vulkan, the same model in BF16, F16 (same bytes) and Q8_0 (what users run). 0.8B: the GGUF of
# /opt3/bench-qwen35 (converted on 29 August); 4B: F16 and Q8_0 made here from the BF16 GGUF with
# llama-quantize 08ffba21e. Prefill 128..4000 (-r 5), decode 255 tokens at depths 759 and 2880 (-r 5).
set -u
L=/opt3/llama.cpp-08ffba21e; G=/opt3/bench-qwen35; O=~/v3-formats; mkdir -p $O
{ date -u; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; (cd $L && git log -1 --format='%h %ci'); df -h $G; } > $O/env-start.txt 2>&1
cmake --build $L/build-cuda -j 24 --target llama-quantize > $O/build-quantize.log 2>&1
for q in F16 Q8_0; do
  [ -f $G/Qwen3.5-4B-$q.gguf ] || $L/build-cuda/bin/llama-quantize $G/Qwen3.5-4B-BF16.gguf $G/Qwen3.5-4B-$q.gguf $q > $O/quantize-4B-$q.log 2>&1
done
sha256sum $G/Qwen3.5-*.gguf > $O/gguf.sha256
for m in 0.8B 4B; do
  for q in BF16 F16 Q8_0; do
    g=$G/Qwen3.5-$m-$q.gguf
    $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/pp-$m-$q-cuda.json 2> $O/pp-$m-$q-cuda.err
    $L/build-vk/bin/llama-bench -m $g -dev Vulkan1 -ngl 99 -p 128,256,512,1024,2048,4000 -n 0 -r 5 -o json > $O/pp-$m-$q-vk.json 2> $O/pp-$m-$q-vk.err
    for d in 759 2880; do
      $L/build-cuda/bin/llama-bench -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d $d -r 5 -o json > $O/tg$d-$m-$q-cuda.json 2> $O/tg$d-$m-$q-cuda.err
      $L/build-vk/bin/llama-bench -m $g -dev Vulkan1 -ngl 99 -p 0 -n 255 -d $d -r 5 -o json > $O/tg$d-$m-$q-vk.json 2> $O/tg$d-$m-$q-vk.err
    done
    echo "$m $q · $(date -u +%T)"
  done
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv > $O/clocks-end.txt
echo ALLDONE-V3
