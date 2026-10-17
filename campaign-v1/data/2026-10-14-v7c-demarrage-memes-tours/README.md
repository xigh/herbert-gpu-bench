# 2026-10-14 — V7c : le démarrage, Herbert, llama.cpp Vulkan et llama.cpp CUDA dans les mêmes tours (4080, bf16)

En relisant #002 : le tableau du démarrage prenait llama.cpp CUDA dans V7 (#551) et Herbert et llama.cpp Vulkan dans V7b (#552, après le nouveau chargeur). Un lecteur demanderait pourquoi pas la même série. Refait dans les mêmes tours, en bf16 (le format du tableau de l'article). Même script que V7b (`_scripts/`, identiques à herbert-gpu-bench 9099655), même arbre (herbert-vk-f16 2ac926d construit sans `f16`, soit le moteur de herbert-vk-bf16 au bit près), llama.cpp 08ffba21e, 5 tours plus un tour 0 écarté, à chaud puis à froid. Lancement : `M=/opt3/models G=/opt3/bench-qwen35 L=/opt3/llama.cpp-08ffba21e F=$HOME/herbert-vk-f16-load O=$HOME/v7c/4080 HDEV=nvidia VKPAT="RTX 4080" CUDA=1 N=5 ENGINES="herbert-bf16 llama-vk-bf16 llama-cuda-bf16"`. 72 démarrages, aucune erreur. Tableaux : `v7c.summary`.

**Condition (`v7c/4080/condition.txt`)** : une autre tâche de beast était en pause (SIGSTOP, volontaire) ; son processus Python gardait 6,4 Go sur la 4080 sans calculer, la carte en P2 avec la mémoire à 10 801 MHz. Contrôle : Herbert et llama.cpp Vulkan retombent sur V7b à 0,02 s près (0.8B à froid 2,12 contre 2,11 et 3,10 contre 3,09 ; 4B à chaud 1,04 contre 1,04 et 1,48 contre 1,47). llama.cpp CUDA démarre un peu plus tôt qu'en V7 (0,76 contre 0,82 s ; 1,21 contre 1,26 s) ; hypothèse non vérifiée : un contexte CUDA déjà ouvert sur la carte. Si c'est cela, le biais est en faveur de llama.cpp CUDA.

| premier token (s), médianes de 5 | 0.8B à froid | 0.8B à chaud | 4B à froid | 4B à chaud |
|---|---|---|---|---|
| Herbert bf16 | 2,12 | 0,61 | 2,56 | 1,04 |
| llama.cpp Vulkan BF16 | 3,10 | 0,77 | 4,97 | 1,48 |
| llama.cpp CUDA BF16 | 0,76 | 0,76 | 1,21 | 1,21 |

Herbert : émission du SPIR-V 2,2 à 2,8 ms, construction par le pilote 1,54 s à froid et 3,7 ms à chaud, poids chargés en 0,30 s (0.8B) et 0,70 s (4B). Les conclusions de #552 tiennent : à chaud, Herbert démarre le premier ; à froid, après llama.cpp CUDA, avant llama.cpp Vulkan.
