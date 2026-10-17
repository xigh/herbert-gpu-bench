#!/usr/bin/env bash
# V2 on g5500 (Windows 11, RTX 3060 Ti, clocks locked by hand), run by Git Bash: herbert-vk-serve
# (9738be7, MSVC) against llama-server b11564 (official archives, CUDA 12.4 and Vulkan), served one after
# the other, timed by v2-client (the Rust copy of _v2_http.py: no Python on this machine). 0.8B and 2B.
set -u
U0=/c/Users/zexig; M=$U0/models; G=$U0/bench-engines/gguf; LB=$U0/bench-engines/llama-b11564
HS=$U0/herbert-vk-bf16-serve/target/release/herbert-vk-serve.exe; CL=$U0/_client/target/release/v2-client.exe
O=$U0/v2-g5500; mkdir -p $O
{ date -u; cat $U0/herbert-vk-bf16-serve/COMMIT; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-start.txt 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu --format=csv -lms 500 > $O/clocks.csv 2>&1 &
SMI=$!
wait_ready() { for i in $(seq 1 $2); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done; return 1; }
for m in 0.8B 2B; do
  for e in herbert llama-cuda llama-vk; do
    case $e in
      herbert) "$HS" $M/Qwen3.5-$m --device nvidia --ctx 4608 --port 8101 --name Qwen3.5-$m > $O/server-$e-$m.log 2>&1 & IMG=herbert-vk-serve.exe; U=http://127.0.0.1:8101; H=$U/;;
      llama-cuda) "$LB/cuda/llama-server.exe" -m $G/Qwen3.5-$m-bf16.gguf -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103 > $O/server-$e-$m.log 2>&1 & IMG=llama-server.exe; U=http://127.0.0.1:8103; H=$U/health;;
      llama-vk) "$LB/vulkan/llama-server.exe" -m $G/Qwen3.5-$m-bf16.gguf -dev Vulkan0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104 > $O/server-$e-$m.log 2>&1 & IMG=llama-server.exe; U=http://127.0.0.1:8104; H=$U/health;;
    esac
    if wait_ready $H 600; then
      "$CL" --prompts $U0/prompts.json --url $U/v1/messages --engine $e --model Qwen3.5-$m --out $O/http-$e-$m.json > $O/client-$e-$m.log 2>&1
      echo "$m $e client exit $? · $(date -u +%T)"; cat $O/client-$e-$m.log
    else
      echo "$m $e: server not ready"; tail -5 $O/server-$e-$m.log
    fi
    taskkill //F //IM $IMG > /dev/null 2>&1; sleep 5
  done
done
kill $SMI 2>/dev/null
date -u > $O/env-end.txt
echo ALLDONE-V2-G5500
