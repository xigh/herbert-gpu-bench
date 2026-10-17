#!/usr/bin/env bash
# V7 on the 4080 of beast (clocks locked), after V6, with vLLM.
until grep -q ALLDONE-V6 $HOME/v6-sweep.out 2>/dev/null; do sleep 60; done
M=/opt3/models G=/opt3/bench-qwen35 L=/opt3/llama.cpp-08ffba21e F=$HOME/herbert-vk-f16-e4901c4 O=$HOME/v7/4080 HDEV=nvidia VKPAT="RTX 4080" CUDA=1 N=5 VL=$HOME/venvs/bench-vllm-cu/bin/vllm bash $HOME/_v7-startup.sh
