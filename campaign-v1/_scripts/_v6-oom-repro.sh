#!/usr/bin/env bash
# After V7 on beast: the llama.cpp CUDA abort of V6 (4B F16, -p 2880 -b 2048 -ub 256 -fa 1, after three other
# settings in the same llama-bench process: "CUDA error: out of memory" in cudaGraphInstantiate, from
# ggml_cuda_graph_evaluate_and_capture). Hypothesis: the CUDA graph cache of ggml-cuda (a map keyed by the first
# node's pointer, evicted only after 10 s unused) grows during the micro-batches of one prefill until the memory
# left by the 4B runs out. Cases, the GPU memory sampled every 100 ms, llama-bench verbose (graph logs):
#   A the failing setting alone in a fresh process      B A with GGML_CUDA_DISABLE_GRAPHS=1
#   C the four settings in V6's order in one process     D A with -ub 512 (the default)
set -u
L=/opt3/llama.cpp-08ffba21e; G=/opt3/bench-qwen35; O=${O:-$HOME/v6-oom}; B=$L/build-cuda/bin/llama-bench; g=$G/Qwen3.5-4B-F16.gguf
until grep -q ALLDONE-V7 $HOME/v7-beast.out 2>/dev/null; do sleep 60; done
mkdir -p $O
{ date -u; (cd $L && git log -1 --format='%h %ci'); nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv; } > $O/env.txt 2>&1
run() { local name=$1; shift
  nvidia-smi --query-gpu=timestamp,memory.used,clocks.mem,pstate --format=csv -lms 100 > $O/$name-mem.csv 2>&1 & local S=$!
  sleep 1; env "$@" > $O/$name.json 2> $O/$name.err; echo "$name exit $? · $(date -u +%T)"; sleep 1; kill $S; sleep 5; }
run A $B -m $g -dev CUDA0 -ngl 99 -p 2880 -n 0 -b 2048 -ub 256 -fa 1 -r 5 -o json -v
run B GGML_CUDA_DISABLE_GRAPHS=1 $B -m $g -dev CUDA0 -ngl 99 -p 2880 -n 0 -b 2048 -ub 256 -fa 1 -r 5 -o json -v
run C $B -m $g -dev CUDA0 -ngl 99 -p 759,2880 -n 0 -b 2048 -ub 256 -fa 0,1 -r 5 -o json -v
run D $B -m $g -dev CUDA0 -ngl 99 -p 2880 -n 0 -b 2048 -ub 512 -fa 1 -r 5 -o json -v
echo ALLDONE-OOM
