#!/usr/bin/env bash
# V1 on a machine measured through Vulkan only (protocol.md): herbert-vk-bf16 2f3ae48 against
# llama.cpp 08ffba21e Vulkan, p3 shape through bench/vs.sh (N = 5 interleaved, cold and warm),
# then with P4=1 the p4 shape (2 880 + 255), N = 5. Usage: _v1-vk.sh OUT LABEL HERBERT_DEVICE LLAMA_DEVICE_PATTERN
set -u
O=$1; LABEL=$2; HDEV=$3; PAT=$4
H=~/herbert-vk-bf16-v1; HV=$H/target/release/herbert-vk-chat
L=~/llama.cpp-08ffba21e; LV=$L/build-vk/bin/llama-bench; G=~/gguf; M=~/models
mkdir -p $O
envs() {
  { date -u; uptime; cat $H/COMMIT; (cd $L && git log -1 --format='%h %ci' && git rev-list --count HEAD); $LV --list-devices
    vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'; uname -r; } > $O/env-$1.txt 2>&1
}
envs start
LDEV=$($LV --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "$PAT" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
[ -n "$LDEV" ] || { echo "no llama.cpp device for '$PAT'"; exit 1; }
echo "llama.cpp device for '$PAT': $LDEV" | tee -a $O/env-start.txt
mkdir -p $O/prompts; cp $H/bench/prompts/p*.txt $O/prompts/
( cd $H && LLCUDA="" LLVK=$LV GGUF=$G MODELS=$M HVBIN=$HV PAIRS="$LDEV:$HDEV:$LABEL" CUDAPAIR="" WARM=1 \
  SIZES="${SIZES:-0.8B 4B}" bench/vs.sh $O/p3 5 ) > $O/p3.out 2>&1
echo "p3 done · $(date -u +%T)"
if [ "${P4:-0}" = 1 ]; then
  ask=$(cat $O/prompts/p4-xlong.txt); mkdir -p $O/p4
  for r in 1 2 3 4 5; do
    for m in ${SIZES:-0.8B 4B}; do
      g=$G/Qwen3.5-$m-BF16.gguf
      $LV -m $g -dev $LDEV -ngl 99 -p 2880 -n 0 -r 1 -o json > $O/p4/r$r-$m-llama-vk-pp.json 2> $O/p4/r$r-$m-llama-vk-pp.err
      $LV -m $g -dev $LDEV -ngl 99 -p 0 -n 255 -d 2880 -r 1 -o json > $O/p4/r$r-$m-llama-vk-tg.json 2> $O/p4/r$r-$m-llama-vk-tg.err
      $HV $M/Qwen3.5-$m --device $HDEV --no-system --max-tokens 256 --warmup 1 --ask "$ask" > $O/p4/r$r-$m-herbert.log 2>&1
    done
    echo "p4 round $r · $(date -u +%T)"
  done
fi
envs end
echo ALLDONE-V1-$LABEL
