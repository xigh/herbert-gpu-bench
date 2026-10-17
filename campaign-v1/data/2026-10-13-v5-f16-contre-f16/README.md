# 2026-10-13 — V5 : f16 contre f16, Herbert f16 contre llama.cpp F16 sur quatre GPU, au banc et au client commun

Le 13 octobre : comparer à format égal. V4 (#545) mettait Herbert bf16 face à llama.cpp F16, deux formats différents ; ce n'est pas une comparaison. Ici, Herbert en f16 (herbert-vk-f16 cc51363, `commits.txt` : poids, activations, cache et matrices coopératives en IEEE f16, accumulateurs f32 ; sur les iGPU AMD sans matrices, le GEMM SIMT en f16 packed, `gemm.h16=1`) contre llama.cpp 08ffba21e en F16 (GGUF de V4), avec Herbert bf16 du même arbre (`--no-default-features --features bf16`, le moteur de herbert-vk-bf16 bit pour bit) dans les mêmes tours. Banc (`_scripts/_v5-f16.sh`) : formes p3/p4 (prefill N puis 255 tokens à la profondeur N), 5 tours, moteurs alternés dans chaque tour, Herbert à chaud, llama-bench `-r 1` aux réglages par défaut (`-b 2048 -ub 512`, flash attention auto). Client : herbert-vk-serve bf16 et f16, llama-server F16, `_v2_http.py` comme V2. 4080 de beast (horloges verrouillées, CUDA et Vulkan, p3 et p4, client), Arc 140V (p3, p4, client), iGPU Raphael et Cezanne (p3, banc seul, comme V4). Tableaux : `v5.summary`.

## Ce que ça dit (« ×1,89 » : Herbert 1,89 fois plus rapide ; sous ×1, llama.cpp devant)

| GPU | modèle | prefill 759 | prefill 2 880 | décodage |
|---|---|---|---|---|
| 4080, contre CUDA F16 | 0.8B | ×1,89 | ×1,47 | +0,5 à +6 % |
| 4080, contre CUDA F16 | 4B | ×1,15 | ×1,01 | +2 à +3 % |
| 4080, contre Vulkan F16 | 0.8B | ×2,04 | ×2,08 | +4 à +8 % |
| 4080, contre Vulkan F16 | 4B | ×1,29 | ×1,17 | +4 à +5 % |
| Arc 140V | 0.8B | ×0,89 | ×0,80 | +4 à +5 % |
| Arc 140V | 4B | ×0,57 | ×0,55 | +1 à +2 % |
| Raphael | 0.8B / 4B | ×1,04 / ×0,61 | | +17 % / +20 % |
| Cezanne | 0.8B / 4B | ×0,96 / ×0,85 | | +11 % / +4 % |

- **Client commun, 4080** : premier token d'Herbert f16 de 1,2 à 3,5 fois plus tôt que le meilleur des deux llama-server F16 (le plus grand écart vers 100 tokens de prompt, le plus petit sur le 4B à 4 105 tokens, ×1,22) ; décodage +3 à +18 %.
- **Client commun, Arc** : Herbert répond le premier jusqu'à une centaine de tokens (×2,0 à ×3,7), à égalité vers 510, derrière au-delà : 4B à 4 105 tokens, 17,5 s contre 10,9 s ; 0.8B à 4 105, 3,8 s contre 3,4 s. Décodage +7 à +10 % sur le 0.8B, à égalité sur le 4B (un point à 9,6 tok/s, 4B à 109 tokens, à revoir).
- **Herbert f16 contre son bf16, mêmes tours** : 4080, prefill 96–98 % et décodage 97–98 % ; Arc, prefill 96–99 %, décodage 93–95 % ; iGPU AMD, prefill +33 à +75 % (calcul en f16), décodage égal.
- **Réglages de llama.cpp** : par défaut ici. Le balayage (V6, puis V6b refait proprement) donne son meilleur réglage ; sur la 4080 en CUDA, le 0.8B gagne jusqu'à 23 % avec `-ub 1024 -b 4096 -fa 1` (V6, à confirmer par V6b).
