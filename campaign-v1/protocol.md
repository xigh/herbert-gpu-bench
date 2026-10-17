# Campagne V1 : les chiffres définitifs de #002 (Vulkan), le 12 octobre 2026

Écrit avant toute mesure, après mes prédictions (`predictions.md`, 9 h 06). But : refaire avec le binaire publié tout ce que #002 cite, au même commit pour les deux llama.cpp, et mesurer ce que la première série ne mesurait pas (le coût fixe de vLLM, les réglages de llama.cpp).

## Ce qui est mesuré

- **Herbert** : `herbert-vk-bf16` au commit `2f3ae48` (bf16 par défaut ; code identique à e692613, celui de wiki #528, aux commentaires près), `cargo build --release --locked` sur chaque machine, défauts par vendeur, sans `--tune` sauf la ligne `host.poll=1` marquée comme telle.
- **llama.cpp** : commit `08ffba21e` (build 11564, 12 octobre 2026), CUDA et Vulkan compilés sur place (beast : CUDA 12.8, sm_89 ; white2 et omnibook : Vulkan), archives officielles `b11564` sur g5500 (Windows : Vulkan et CUDA 12.4). `llama-bench`, GGUF BF16, `-ngl 99`, réglages par défaut (micro-lot 512, flash attention automatique), plus un balayage du micro-lot et de flash attention sur la 4080.
- **vLLM** : 0.28.0 (torch 2.13.0+cu130), son banc de latence, `--batch-size 1`, bf16, 2 tours de chauffe, 5 mesures. Prefill = longueur / latence avec un token généré ; décodage = 254 / (latence à 255 tokens − latence à 1). Coût fixe : la même latence avec un prompt de 1, 16 et 64 tokens.
- **Une requête à la fois** partout. Médianes de N = 5 tours alternés (Herbert et llama.cpp) ; vLLM non alterné (son démarrage est trop long).

## Les formes

- Prefill selon la longueur : 128, 256, 512, 1 024, 2 048, 4 000 tokens, un token généré (comme wiki #528).
- p3 : prefill de 759 tokens puis 255 générés à la profondeur 759 ; p4 : 2 880 puis 255 à 2 880 (comme wiki #510).
- Tokens contre l'oracle CPU (`herbert-bf16-chat`), avec les 5 meilleurs logits (`HERBERT_VK_TOP=5`), comme wiki #540.

## Les machines

| machine | GPU | pilote | mesures |
|---|---|---|---|
| beast | RTX 4080 16 Go, SM 2 805 MHz et mémoire 11 201 MHz verrouillées (12 octobre) | NVIDIA 595.99.02 | tout, quatre moteurs |
| beast | iGPU Ryzen 9 9900X (RADV Raphael) | Mesa | p3, Herbert contre llama.cpp Vulkan |
| white2 | iGPU Ryzen 7 5700G (RADV Renoir, « Cezanne ») | Mesa 25.2.8 | p3, Herbert contre llama.cpp Vulkan |
| omnibook | Arc 140V (Core Ultra 7 258V) | Mesa (ANV) | p3 et p4, Herbert contre llama.cpp Vulkan |
| g5500 | RTX 3060 Ti 8 Go, Windows 11 | NVIDIA 591.86 | p3 et p4, 0.8B et 2B (le 4B ne tient pas), Herbert contre llama.cpp Vulkan et CUDA |

Horloges verrouillées sur la 4080 de beast seulement ; ailleurs, relevées au début et à la fin quand l'outil le permet. Chaque log complet est gardé et déposé au wiki (`raw/data/2026-10-12-v1-*`).

## Ce qui n'est pas mesuré

Pas de vLLM hors beast. Pas de lot de requêtes. Pas de modèle MoE. La 4080 de white2 n'est pas remesurée (même carte que beast, horloges non verrouillées).
