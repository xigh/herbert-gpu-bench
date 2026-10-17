# herbert-gpu-bench

Reproducible GPU benchmarks of [herbert-vk-bf16](https://github.com/xigh/herbert-vk-bf16) and [herbert-vk-f16](https://github.com/xigh/herbert-vk-f16) against llama.cpp (Vulkan and CUDA) and vLLM, on Qwen3.5 0.8B, 2B and 4B, one request at a time, at equal format (bf16 against bf16, f16 against f16), without any quantization: the protocol, the predictions written before measuring, the dated journal, the exact scripts and the complete raw logs. Companion to the article « On descend d'un cran : #002 le même compilateur sur un GPU » on philippe-anel.fr ([FR](https://philippe-anel.fr/fr/blog/2026/10/18/on-descend-2-vulkan/), [EN](https://philippe-anel.fr/en/blog/2026/10/18/on-descend-2-vulkan/)).

The question behind it: Herbert writes its compute kernels at start-up, as SPIR-V, for the exact shapes of the model and for the GPU in front of it. Does that hold against the reference GPU engines on consumer cards, and where does it not?

## Layout

```
campaign-v1/protocol.md      what is measured, how, on which machine (written before any measure)
campaign-v1/predictions.md   the predictions, written before the definitive campaign
campaign-v1/journal.md       dated journal of the campaign: anomalies, errors found, decisions
campaign-v1/_scripts/        the exact drivers and summary scripts (V1 to V8)
campaign-v1/_client/         the common HTTP client in Rust, for a machine without Python (g5500)
campaign-v1/logs/            build and download logs of g5500
campaign-v1/data/            the raw data, one folder per measure, with its own scripts and summaries
```

`campaign-v1/data/README.md` links every section of the article to its folder. The documents of the campaign are in French, like the article; the logs are complete, never filtered nor edited. Times are UTC. The « #5xx » numbers met in the files refer to private working notes: every folder reads without them.

Published as a single commit without history. Private LAN addresses were replaced by machine names, and a few local paths and task names by neutral ones; three Windows minidumps of the 3060 Ti's crashes are left out (their names and SHA-256 prefixes stay in `data/2026-10-12-g5500-tdr-pilote-616/README.md`). The measures, the logs' numbers and the scripts' logic are untouched.

## Machines and GPUs

| machine | GPU | driver | measures |
|---|---|---|---|
| beast | NVIDIA RTX 4080 16 GB, SM 2 805 MHz and memory 11 201 MHz locked | NVIDIA 595.99.02 | everything, four engines |
| beast | iGPU of the Ryzen 9 9900X (RADV, Raphael) | Mesa 25.2.8 | Herbert against llama.cpp Vulkan |
| white2 | iGPU of the Ryzen 7 5700G (RADV, Renoir) | Mesa 25.2.8 | Herbert against llama.cpp Vulkan |
| omnibook | Intel Arc 140V (Core Ultra 7 258V) | Mesa (ANV) | Herbert against llama.cpp Vulkan |
| g5500 | NVIDIA RTX 3060 Ti 8 GB, Windows 11 | NVIDIA 591.86, then 616.92 | 0.8B and 2B (the 4B does not fit), Herbert against llama.cpp Vulkan and CUDA |

Clocks are locked on the 4080 only; elsewhere they are logged at the start and at the end of each run. The 3060 Ti drops its clocks under load with both drivers (`data/2026-10-12-g5500-tdr-pilote-616/`): its numbers are given with that caveat, bf16 against bf16 only.

Other GPUs are untested, and the default tunings of the engines may not suit them. Discrete AMD Radeon cards will come in a later version, with their own article.

## Engines

| engine | version | format |
|---|---|---|
| Herbert | [herbert-vk-bf16 v0.1.0](https://github.com/xigh/herbert-vk-bf16/tree/v0.1.0) and [herbert-vk-f16 v0.1.0](https://github.com/xigh/herbert-vk-f16/tree/v0.1.0), built on each machine with `cargo build --release --locked` (`--features serve` for the HTTP measures), defaults per vendor | safetensors, bf16 or f16 |
| llama.cpp | commit `08ffba21e` (build 11564, 12 October 2026), CUDA 12.8 and Vulkan built on the Linux machines, official `b11564` archives on g5500 | GGUF BF16 and F16, `-ngl 99` |
| vLLM | 0.28.0, torch 2.13.0+cu130, beast only | bf16 and float16 |

v0.1.0 of each engine is the copy with the new loader (`data/2026-10-13-chargeur-herbert-vk/`). The earlier measures name the commit they ran in their logs: herbert-vk-bf16 `2f3ae48` (V1), `9738be7` (the same engine plus the HTTP server, V2), herbert-vk-f16 `cc51363` and `e4901c4` (V5 to V7). Between those commits and v0.1.0, only the loading of the weights, the server and a start-up log line change: the emitter, the kernels and the defaults (`src/spv`, `src/vk/kernels.rs`, `src/vk/prefill.rs`, `src/vk/tuning.rs`) are identical, and the weights on the card are identical to the bit (checked on the five best candidates of every token). The start-up tables (V7b, V7c) and the ablation (V8) were measured with v0.1.0 itself.

## Two instruments

- **Each engine's own bench** (`herbert-vk-chat`, `llama-bench`, `vllm bench latency`): prefill and decode as each engine counts them, Herbert and llama.cpp interleaved in N = 5 rounds, medians.
- **The common client** (`_scripts/_v2_http.py`, `_client/`): the same HTTP client for every engine (the Anthropic Messages API, streamed, greedy, one request at a time), so the first token and the decode are timed from the user's side, fixed costs included.

## Reproduce

`campaign-v1/protocol.md` gives the settings; each `_scripts/_v*.sh` starts with the variables it needs (engine trees, models, prompts, output folder) and each `_v*_summary.py` rebuilds the tables from the logs. Every folder of `data/` keeps the copy of the scripts it was run with and its launch line in its `README.md`.

## License

MIT, see `LICENSE`.
