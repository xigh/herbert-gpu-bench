#!/usr/bin/env bash
# V1 on g5500 (Windows 11, RTX 3060 Ti 8 GB), run by Git Bash: herbert-vk-bf16 2f3ae48 (MSVC build)
# against the official llama.cpp b11564 archives (Vulkan and CUDA 12.4), p3 (759 + 255 at 759) then
# p4 (2 880 + 255 at 2 880), 0.8B and 2B (the 4B does not fit in 8 GB), N = 5 interleaved, herbert cold
# and warm. Same file names as bench/vs.sh. Every output kept in full under $O.
# Under _watchdog-win.sh (90 °C, or the clock log frozen), each measurement after a cooldown to 70 °C.
set -u
U=/c/Users/zexig; H=$U/herbert-vk-bf16-v1; HV=$H/target/release/herbert-vk-chat.exe
LV=$U/bench-engines/llama-b11564/vulkan/llama-bench.exe; LC=$U/bench-engines/llama-b11564/cuda/llama-bench.exe
G=$U/bench-engines/gguf; M=$U/models; O=${O:-$U/v1-g5500}; mkdir -p $O/p3 $O/p4 $O/prompts
cp $H/bench/prompts/p*.txt $O/prompts/
envs() { { date -u; cat $H/COMMIT; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv
  "$LV" --list-devices; "$LC" --list-devices; sha256sum "$HV" "$LV" "$LC"; } > $O/env-$1.txt 2>&1; }
envs start
# clocks every 0.5 s for the whole pass (the lock may not hold above P2 on Windows)
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,fan.speed --format=csv -lms 500 > $O/clocks.csv 2>&1 &
SMI=$!
. "$(dirname "$0")/_watchdog-win.sh"; watchdog $O $$ &
WD=$!
for shape in p3 p4; do
  case $shape in p3) N=759; ask=$(cat $O/prompts/p3-long.txt);; p4) N=2880; ask=$(cat $O/prompts/p4-xlong.txt);; esac
  for r in 1 2 3 4 5; do
    for m in 0.8B 2B; do
      g=$G/Qwen3.5-$m-bf16.gguf; t=$O/$shape/r$r-$m
      cooldown $O
      "$LC" -m $g -dev CUDA0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-cuda-nv-pp.json 2> $t-llama-cuda-nv-pp.err
      cooldown $O
      "$LC" -m $g -dev CUDA0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-cuda-nv-tg.json 2> $t-llama-cuda-nv-tg.err
      cooldown $O
      "$LV" -m $g -dev Vulkan0 -ngl 99 -p $N -n 0 -r 1 -o json > $t-llama-vk-nv-pp.json 2> $t-llama-vk-nv-pp.err
      cooldown $O
      "$LV" -m $g -dev Vulkan0 -ngl 99 -p 0 -n 255 -d $N -r 1 -o json > $t-llama-vk-nv-tg.json 2> $t-llama-vk-nv-tg.err
      cooldown $O
      "$LV" -m $g -dev Vulkan0 -ngl 99 -p $N -n 0 -r 1 --no-warmup -o json > $t-llama-vk-nv-cold-pp.json 2> $t-llama-vk-nv-cold-pp.err
      cooldown $O
      "$HV" $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 256 --ask "$ask" > $t-vk-nv-default.log 2>&1
      cooldown $O
      "$HV" $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $t-vk-nv-warm.log 2>&1
    done
    echo "$shape round $r · $(date -u +%T)"
  done
done
envs end

# Tokens against the CPU oracle (herbert-jit-bf16 v0.1.0, MSVC build of C1), 5 best logits on the GPU, as compare.sh.
CPU=$U/c1/pub/target/release/herbert-bf16-chat.exe
for m in 0.8B 2B; do
  mkdir -p $O/cmp-$m
  for p in $O/prompts/p*.txt; do
    name=$(basename "$p" .txt); think="--no-think"; max=200
    case $name in p3-*) think=""; max=256;; esac
    "$CPU" $M/Qwen3.5-$m --no-system $think --max-tokens $max --ask "$(cat "$p")" > $O/cmp-$m/$name-cpu.log 2>&1
    HERBERT_VK_TOP=5 "$HV" $M/Qwen3.5-$m --device nvidia --no-system $think --max-tokens $max --ask "$(cat "$p")" > $O/cmp-$m/$name-vk-nvidia-top.log 2>&1
    echo "tokens $m $name · $(date -u +%T)"
  done
done
kill $SMI 2>/dev/null
touch $O/DONE
if [ -f $O/ABORT ]; then echo STOPPED-V1-G5500; else echo ALLDONE-V1-G5500; fi
