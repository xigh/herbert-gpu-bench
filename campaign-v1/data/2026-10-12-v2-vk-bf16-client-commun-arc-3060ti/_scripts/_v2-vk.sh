#!/usr/bin/env bash
# V2 on a machine measured through Vulkan only: herbert-vk-serve (9738be7) against llama-server
# 08ffba21e Vulkan, served one after the other, timed by the same client as on beast (_v2_http.py,
# Anthropic Messages API, streamed, greedy, 128 tokens, no prefix cache, the same passages).
# Usage: _v2-vk.sh OUT HERBERT_DEVICE LLAMA_DEVICE_PATTERN [SIZES]
set -u
O=$1; HDEV=$2; PAT=$3; SIZES=${4:-0.8B 4B}
M=~/models; G=~/gguf; L=~/llama.cpp-08ffba21e; HS=~/herbert-vk-bf16-serve/target/release/herbert-vk-serve
mkdir -p $O
LDEV=$($L/build-vk/bin/llama-server --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "$PAT" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
[ -n "$LDEV" ] || { echo "no llama.cpp device for '$PAT'"; exit 1; }
{ date -u; uname -r; cat ~/herbert-vk-bf16-serve/COMMIT; (cd $L && git log -1 --format='%h %ci'); $L/build-vk/bin/llama-server --list-devices; echo "llama device $LDEV"; } > $O/env-start.txt 2>&1
wait_ready() { for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
stop() { kill -TERM -- -$1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 1; done; kill -KILL -- -$1 2>/dev/null; sleep 3; }
for m in $SIZES; do
  for e in herbert llama-vk; do
    case $e in
      herbert) setsid $HS $M/Qwen3.5-$m --device $HDEV --ctx 4608 --port 8101 --name Qwen3.5-$m > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8101; H=$U/;;
      llama-vk) setsid $L/build-vk/bin/llama-server -m $G/Qwen3.5-$m-BF16.gguf -dev $LDEV -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & P=$!; U=http://127.0.0.1:8104; H=$U/health;;
    esac
    if wait_ready $H 600; then
      python3 ~/_v2_http.py --prompts ~/prompts.json --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
      echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
    else
      echo "$m $e: server not ready"; tail -5 $O/server-$e-$m.log
    fi
    stop $P
  done
done
date -u > $O/env-end.txt
echo ALLDONE-V2
