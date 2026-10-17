# 2026-10-07 — herbert-vk-bf16 contre vLLM et llama.cpp : le prefill selon la longueur du prompt

Suite de #513 (récapitulatif à plusieurs longueurs). 4080 de beast à 2 805 MHz verrouillés (`clocks-*.txt`), `Vulkan0` = la 4080 (`devices.txt`). herbert : build `bf16` à 5922789, défauts NVIDIA, à chaud (`--warmup 1`), un token généré, 5 tours, médiane ; prompts tirés de p4-xlong doublé et coupés pour 128, 256, 513, 1 024, 2 048 et 4 001 tokens (gabarit compris, `pp1-beast/prompts/`). llama.cpp c841aeeb8 : `llama-bench -p 128,…,4000 -n 0 -r 5` (CUDA0, Vulkan0). vLLM : `vllm bench latency --input-len N --output-len 1`, 2 tours de chauffe, 5 mesures, prefill = N / latence (latence de bout en bout de son banc, ordonnanceur et premier token compris).

## Prefill, tok/s

| modèle | moteur | 128 | 256 | 512 | 1 024 | 2 048 | 4 000 |
|---|---|---|---|---|---|---|---|
| 0.8B | herbert | 26 634 | 41 798 | 48 722 | 56 185 | 56 006 | 54 146 |
| 0.8B | vLLM | 9 376 | 19 132 | 36 888 | 53 147 | 53 438 | 52 487 |
| 0.8B | llama.cpp CUDA | 18 800 | 25 450 | 29 983 | 30 351 | 30 714 | 30 275 |
| 0.8B | llama.cpp Vulkan | 7 171 | 10 880 | 12 084 | 12 057 | 12 131 | 11 897 |
| 4B | herbert | 3 808 | 6 518 | 9 382 | 10 785 | 10 674 | 10 441 |
| 4B | vLLM | 5 516 | 8 605 | 9 790 | 10 360 | 10 909 | 10 778 |
| 4B | llama.cpp CUDA | 5 189 | 7 246 | 7 833 | 7 899 | 7 941 | 7 770 |
| 4B | llama.cpp Vulkan | 1 806 | 2 512 | 2 414 | 2 420 | 2 425 | 2 369 |

Ids de herbert identiques d'un tour à l'autre (vérifié à 128 et 4 000 tokens, 0.8B et 4B). Le 0.8B de herbert est devant vLLM à toutes les longueurs (×2,8 à 128 tokens, +3 à +6 % de 1 024 à 4 000). Le 4B est derrière vLLM sous 1 024 tokens (−31 % à 128, −24 % à 256, −4 % à 512) et même derrière llama.cpp CUDA à 128 et 256 ; devant à 1 024 (+4 %), à −2 % et −3 % à 2 048 et 4 000.

Fichiers : `pp1-beast/` (sorties complètes de herbert, JSON de llama-bench et de vLLM, prompts), `pp1.summary`, `pp1.out`, `commits.txt`, `_pp.py`, `_pp1.sh`.
