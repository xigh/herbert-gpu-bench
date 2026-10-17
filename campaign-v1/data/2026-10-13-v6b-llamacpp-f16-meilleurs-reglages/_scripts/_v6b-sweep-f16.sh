#!/usr/bin/env bash
# V6b, after the OOM reproduction on beast (4080, clocks locked): llama.cpp 08ffba21e in F16 at its best settings,
# done again the way of V5. V6 ran the whole grid in one llama-bench process with -r 5: one abort (4B CUDA, CUDA
# graph OOM) took the rest of the grid with it, and its decode figures (265 tok/s, samples 157-310) disagreed with V5
# (314, stable) for the same binary and settings, the memory clock at 10 801 MHz half of the time. Here every setting
# is its own process, -r 1, N = 5 rounds, Herbert f16 (herbert-vk-f16 e4901c4) in the same rounds, clocks every 0.5 s.
# Grid: prefill 759 and 2 880 tokens, -b 2048/4096, -ub 256/512/1024/2048, -fa 0/1; decode of 255 tokens at depth
# 759 and 2 880, -fa 0/1; CUDA0 and the 4080 through Vulkan; 0.8B and 4B.
set -u
L=/opt3/llama.cpp-08ffba21e; G=/opt3/bench-qwen35; M=/opt3/models; F=$HOME/herbert-vk-f16-e4901c4; O=${O:-$HOME/v6b}
until grep -q ALLDONE-OOM $HOME/v6-oom.out 2>/dev/null; do sleep 60; done
mkdir -p $O/bench
VK=$($L/build-vk/bin/llama-bench --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "RTX 4080" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
{ date -u; cat $F/COMMIT; (cd $L && git log -1 --format='%h %ci'); echo "Vulkan device $VK"; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; sha256sum $F/target/release/herbert-vk-chat $G/Qwen3.5-*-F16.gguf; } > $O/env-start.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,memory.used --format=csv -lms 500 > $O/clocks.csv 2>&1 & SMI=$!
for r in 1 2 3 4 5; do
  for m in 0.8B 4B; do
    g=$G/Qwen3.5-$m-F16.gguf
    for N in 759 2880; do
      case $N in 759) ask=$(cat $F/bench/prompts/p3-long.txt);; 2880) ask=$(cat $F/bench/prompts/p4-xlong.txt);; esac
      t=$O/bench/r$r-$m-p$N
      $F/target/release/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $t-herbert-f16.log 2>&1
      for be in cuda vk; do
        case $be in cuda) B=$L/build-cuda/bin/llama-bench; D=CUDA0;; vk) B=$L/build-vk/bin/llama-bench; D=$VK;; esac
        for b in 2048 4096; do for ub in 256 512 1024 2048; do for fa in 0 1; do
          $B -m $g -dev $D -ngl 99 -p $N -n 0 -b $b -ub $ub -fa $fa -r 1 -o json > $t-$be-b$b-ub$ub-fa$fa-pp.json 2> $t-$be-b$b-ub$ub-fa$fa-pp.err
        done; done; done
        for fa in 0 1; do
          $B -m $g -dev $D -ngl 99 -p 0 -n 255 -d $N -fa $fa -r 1 -o json > $t-$be-fa$fa-tg.json 2> $t-$be-fa$fa-tg.err
        done
      done
    done
  done
  echo "round $r · $(date -u +%T)"
done
kill $SMI
{ date -u; nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-end.txt 2>&1
echo ALLDONE-V6B
