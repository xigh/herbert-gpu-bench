# 2026-10-13 — V6b : llama.cpp F16 à ses meilleurs réglages contre Herbert f16, sur la 4080 ; le premier balayage écarté ; le plantage CUDA non reproduit

Suite de V5 (#549, llama.cpp aux réglages par défaut). Ma question : Herbert f16 ×1,9 sur le 0.8B, est-ce que llama.cpp est désavantagé par ses réglages ? Le balayage micro-lot / flash attention fait avant l'avait été sur le GGUF BF16.

**Premier balayage, écarté** (`v6-premier-balayage-ecarte/`, `_scripts/_v6-sweep-f16.sh`) : toute la grille dans un seul processus llama-bench par modèle et backend, `-r 5`. Deux défauts : (1) sur le 4B en CUDA, abandon au 4e réglage (`-p 2880 -b 2048 -ub 256 -fa 1`) sur « CUDA error: out of memory » dans `cudaGraphInstantiate`, qui emporte le reste de la grille ; (2) décodage de llama.cpp CUDA à 252–272 tok/s, échantillons de 157 à 310, contre 314 stables en V5 avec le même binaire, la mémoire à 10 801 MHz la moitié du temps.

**Le plantage, non reproduit** (`oom-repro/`, `_scripts/_v6-oom-repro.sh`, carte seule, mémoire relevée toutes les 100 ms) : A le réglage fautif seul, B le même sans graphes CUDA, C les quatre réglages dans l'ordre du balayage dans un seul processus, D avec `-ub 512` : tous passent, la mémoire monte au plus à 8 743–8 999 Mio sur 16 376. Le cache de graphes CUDA de ggml (une table sans borne, éviction après 10 s sans usage) n'explique donc pas un manque de mémoire ici ; l'hypothèse restante est qu'un autre processus occupait la carte à ce moment-là, sans preuve (la mémoire n'était pas relevée). Pas de signalement à llama.cpp.

**V6b** (`v6b-4080/`, `_scripts/_v6b-sweep-f16.sh`) : chaque réglage dans son propre processus, `-r 1`, 5 tours, Herbert f16 (herbert-vk-f16 e4901c4) dans les mêmes tours, horloges relevées. Grille : prefill 759 et 2 880 tokens, `-b 2048/4096`, `-ub 256/512/1024/2048`, `-fa 0/1` ; décodage aux mêmes profondeurs, `-fa 0/1` ; CUDA et Vulkan. Aucune sortie vide ; dispersion du meilleur réglage ≤ 2 %. Tableaux : `v6b.summary`.

| modèle | prefill | Herbert f16 contre le meilleur llama.cpp CUDA | contre le meilleur Vulkan | meilleur réglage de llama.cpp |
|---|---|---|---|---|
| 0.8B | 759 | ×1,54 (51 374 contre 33 327) | ×1,78 | `-ub 2048 -fa 1` (CUDA), `-b 4096 -ub 1024 -fa 1` (Vulkan) |
| 0.8B | 2 880 | ×1,42 | ×1,87 | `-b 4096 -ub 1024 -fa 1` |
| 4B | 759 | ×1,01 | ×1,12 | `-ub 1024 -fa 1` |
| 4B | 2 880 | **×0,98** (10 457 contre 10 672) | ×1,16 | `-b 4096 -ub 1024 -fa 1` (CUDA), `-ub 512 -fa 1` (Vulkan) |

Le réglage par défaut (`-b 2048 -ub 512`) coûte à llama.cpp CUDA jusqu'à 16–19 % sur le 0.8B à 759 tokens et le 4B à 759 ; `-ub 1024` et flash attention sont presque toujours son meilleur choix. Décodage : Herbert f16 de +0,4 à +8,5 % (CUDA 0.8B 2 880 : +0,4 %).

**Horloge mémoire et CUDA** (`pstate-cuda-vk/`, contrôle ajouté, 0.8B, décodage) : sous CUDA, la 4080 passe en P2, mémoire à 10 801 MHz (13 relevés sur 16), malgré le verrouillage à 11 201 ; sous Vulkan, P0 à 11 201. Même effet que sous Windows (#543). Au décodage, limité par la bande passante, les moteurs CUDA (llama.cpp CUDA, vLLM) ont 3,6 % de bande passante mémoire de moins que les moteurs Vulkan (Herbert, llama.cpp Vulkan) : l'avance d'Herbert au décodage sur llama.cpp CUDA (+0,4 à +5,8 %) et l'écart avec vLLM (±3 %, #541) sont de cet ordre.
