#!/usr/bin/env bash
# V7: start-up, from the launch of the server to its first token, at equal format: Herbert bf16 / llama.cpp BF16,
# Herbert f16 / llama.cpp F16 (llama.cpp's shaders are compiled to SPIR-V when llama.cpp is built and embedded;
# Herbert emits its SPIR-V at start-up without an external compiler; both then have the driver build the pipelines).
# Cold: the driver's shader caches disabled (__GL_SHADER_DISK_CACHE=0, MESA_SHADER_CACHE_DISABLE=true,
# CUDA_CACHE_DISABLE=1); warm: the caches on, after a first start. The weight files are read once before each start
# (page cache), so the disk is not measured. N rounds, the engines interleaved. Herbert's log splits the SPIR-V
# emission from the driver's build (herbert-vk-f16 e4901c4). With VL (the vllm executable), vLLM too, in bf16 and in
# float16 (--dtype), as served in V2; vLLM compiles at start-up as well (torch.compile, Triton, CUDA graphs): cold, its
# caches (VLLM_CACHE_ROOT, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR) point to an empty directory, removed after the
# start. Settings as _v5-f16.sh: M G L F O HDEV VKPAT CUDA SIZES, N, VL.
set -u
: "${M:?}" "${G:?}" "${L:?}" "${F:?}" "${O:?}" "${HDEV:?}" "${VKPAT:?}"
CUDA=${CUDA:-0}; SIZES=${SIZES:-0.8B 4B}; N=${N:-5}; HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
mkdir -p $O/logs
( cd $F && cargo build --release --locked --features serve ) > $O/build-f16.log 2>&1 || { echo "build f16 failed"; exit 1; }
( cd $F && cargo build --release --locked --no-default-features --features "bf16 serve" --target-dir target-bf16 ) > $O/build-bf16.log 2>&1 || { echo "build bf16 failed"; exit 1; }
LDEV=$($L/build-vk/bin/llama-server --list-devices 2>&1 | grep -E '^ *Vulkan[0-9]+:' | grep -i -- "$VKPAT" | head -1 | sed -E 's/^ *(Vulkan[0-9]+):.*/\1/')
{ date -u; uname -r; cat $F/COMMIT; (cd $L && git log -1 --format='%h %ci'); echo "llama.cpp Vulkan device $LDEV"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-start.txt 2>&1
COLD="__GL_SHADER_DISK_CACHE=0 MESA_SHADER_CACHE_DISABLE=true CUDA_CACHE_DISABLE=1"
E="herbert-bf16 llama-vk-bf16 herbert-f16 llama-vk-f16"; [ "$CUDA" = 1 ] && E="$E llama-cuda-bf16 llama-cuda-f16"; [ -n "${VL:-}" ] && E="$E vllm-bf16 vllm-f16"
[ -n "${ENGINES:-}" ] && E=$ENGINES   # a subset, e.g. Herbert alone after a change of its loader
for mode in warm cold; do
  for r in $(seq 0 $N); do   # round 0: fills the caches in warm mode, discarded in both
    for m in $SIZES; do
      for e in $E; do
        fmt=${e##*-}; GG=$G/Qwen3.5-$m-$(echo $fmt | tr a-z A-Z).gguf
        case $e in
          herbert-bf16) C="$F/target-bf16/release/herbert-vk-serve $M/Qwen3.5-$m --device $HDEV --ctx 4608 --port 8101 --name Qwen3.5-$m"; U=http://127.0.0.1:8101; H=$U/; W="$M/Qwen3.5-$m/*.safetensors";;
          herbert-f16) C="$F/target/release/herbert-vk-serve $M/Qwen3.5-$m --device $HDEV --ctx 4608 --port 8102 --name Qwen3.5-$m"; U=http://127.0.0.1:8102; H=$U/; W="$M/Qwen3.5-$m/*.safetensors";;
          llama-vk-*) C="$L/build-vk/bin/llama-server -m $GG -dev $LDEV -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8104"; U=http://127.0.0.1:8104; H=$U/health; W=$GG;;
          vllm-*) D=bfloat16; [ $fmt = f16 ] && D=float16
            C="VLLM_USE_FLASHINFER_SAMPLER=0 $VL serve $M/Qwen3.5-$m --dtype $D --max-model-len 4608 --max-num-seqs 8 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --served-model-name Qwen3.5-$m --default-chat-template-kwargs '{\"enable_thinking\": true}' --port 8105"; U=http://127.0.0.1:8105; H=$U/health; W="$M/Qwen3.5-$m/*.safetensors";;
          llama-cuda-*) C="$L/build-cuda/bin/llama-server -m $GG -dev CUDA0 -ngl 99 -c 4608 -np 1 --no-cache-prompt -a Qwen3.5-$m --port 8103"; U=http://127.0.0.1:8103; H=$U/health; W=$GG;;
        esac
        cat $W > /dev/null
        TMPC=""; if [ $mode = cold ]; then TMPC=$(mktemp -d $O/cold-cache.XXXXXX); C="env $COLD VLLM_CACHE_ROOT=$TMPC/vllm TRITON_CACHE_DIR=$TMPC/triton TORCHINDUCTOR_CACHE_DIR=$TMPC/inductor $C"; fi
        res=$(python3 $HERE/_v7_startup.py --cmd "$C" --health $H --msg $U/v1/messages --model Qwen3.5-$m --log $O/logs/$mode-r$r-$m-$e.log)
        echo "{\"mode\": \"$mode\", \"round\": $r, \"model\": \"$m\", \"engine\": \"$e\", \"result\": $res}" >> $O/startup.jsonl
        [ -n "$TMPC" ] && rm -rf -- "$TMPC"
        sleep 2
      done
    done
    echo "$mode round $r · $(date -u +%T)"
  done
done
date -u > $O/env-end.txt
echo ALLDONE-V7
