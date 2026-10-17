#!/usr/bin/env bash
# V5 on beast: the 4080 (locked clocks, CUDA and Vulkan, bench p3/p4 and client), then the Raphael iGPU (bench p3), one after the other.
set -u
F=$HOME/herbert-vk-f16-cc51363
M=/opt3/models G=/opt3/bench-qwen35 L=/opt3/llama.cpp-08ffba21e F=$F O=$HOME/v5/4080 HDEV=nvidia VKPAT="RTX 4080" CUDA=1 SHAPES="759 2880" CLIENT=1 PROMPTS=/opt3/bench-qwen35/prompts.json bash $HOME/_v5-f16.sh
M=/opt3/models G=/opt3/bench-qwen35 L=/opt3/llama.cpp-08ffba21e F=$F O=$HOME/v5/raphael HDEV=amd VKPAT=RAPHAEL CUDA=0 SHAPES=759 CLIENT=0 bash $HOME/_v5-f16.sh
echo ALLDONE-V5-BEAST
