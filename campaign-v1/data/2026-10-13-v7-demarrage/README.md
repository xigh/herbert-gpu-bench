# 2026-10-13 — V7 : le démarrage, du lancement du serveur au premier token — Herbert, llama.cpp (CUDA, Vulkan) et vLLM, à froid et à chaud, sur la 4080

Demande d'un lecteur après #001, que j'ai reprise le 13 octobre : combien de temps pour démarrer ? llama.cpp compile ses shaders en SPIR-V quand on compile llama.cpp (glslc) et les range dans le binaire, son code CUDA aussi à la compilation (nvcc) ; Herbert écrit son SPIR-V au démarrage, sans compilateur externe ; vLLM compile au démarrage (torch.compile, Triton, graphes CUDA). Les deux moteurs Vulkan font ensuite construire leurs pipelines par le pilote. herbert-vk-f16 e4901c4 sépare dans son log le temps d'émission du SPIR-V du temps de construction par le pilote.

Mesure (`_scripts/_v7-startup.sh`, `_v7_startup.py`) : lancement du serveur, interrogation de sa route de santé toutes les 5 ms (« prêt »), puis une requête d'un token (« premier token »), vu du client. À format égal : Herbert bf16, llama.cpp BF16 (CUDA, Vulkan), vLLM `--dtype bfloat16` ; Herbert f16, llama.cpp F16, vLLM `--dtype float16`. À chaud : caches du pilote et de vLLM en place (un tour 0 écarté les remplit). À froid : `__GL_SHADER_DISK_CACHE=0 MESA_SHADER_CACHE_DISABLE=true CUDA_CACHE_DISABLE=1`, et pour vLLM `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` vers un dossier vide, supprimé après. Fichiers de poids lus avant chaque lancement (cache de pages : le disque n'est pas mesuré). 5 tours, moteurs alternés, 0.8B et 4B. Tableaux : `v7.summary`.

| premier token (s) | 0.8B à froid | 0.8B à chaud | 4B à froid | 4B à chaud |
|---|---|---|---|---|
| Herbert bf16 | 2,60 | 1,08 | 5,94 | 4,40 |
| llama.cpp Vulkan BF16 | 3,13 | 0,79 | 5,03 | 1,47 |
| llama.cpp CUDA BF16 | 0,82 | 0,82 | 1,26 | 1,26 |
| vLLM bf16 | 89,5 | 37,8 | 94,5 | 41,5 |
| Herbert f16 | 4,80 | 3,59 | 20,25 | 19,01 |
| llama.cpp Vulkan F16 | 3,13 | 0,77 | 4,90 | 1,44 |
| llama.cpp CUDA F16 | 0,81 | 0,81 | 1,27 | 1,26 |
| vLLM float16 | 98,8 | 38,1 | 106,9 | 42,3 |

- **L'émission du SPIR-V d'Herbert coûte 1,6 à 3,5 ms** (38 pipelines, 379 Kio). La construction par le pilote : 1,2 à 1,5 s à froid, 2,7 ms à chaud (cache du pilote). Le reste du démarrage d'Herbert est le chargement des poids : 0,8 s (0.8B) et 4,1 s (4B) en bf16 ; **3,3 s et 18,7 s en f16**, la conversion bf16 → f16 au chargement, sur l'hôte, à revoir.
- **llama.cpp CUDA démarre le plus vite** (0,8 et 1,3 s, sans différence froid / chaud : son code est compilé d'avance pour la carte). llama.cpp Vulkan : prêt en 2,1 s à froid sur le 0.8B, premier token à 3,1 s (une partie de ses pipelines se construit à la première requête) ; 0,8 s à chaud.
- **Herbert contre llama.cpp Vulkan** : à froid, premier token plus tôt sur le 0.8B (2,6 contre 3,1 s), plus tard sur le 4B (5,9 contre 5,0 s, le chargement des poids) ; à chaud, plus tard (1,1 contre 0,8 s ; 4,4 contre 1,5 s). Le temps d'Herbert est dans le chargement des poids, pas dans l'émission.
- **vLLM** : 38 à 42 s à chaud, 89 à 107 s à froid. D'après ses journaux (`logs/*-r1-*-vllm-bf16.log`) : à froid, `torch.compile` 9,95 s (0.8B) et 13,81 s (4B), puis un premier passage de profilage de 34,5 à 34,9 s (0,10 à 0,23 s à chaud : ses noyaux compilés à la volée, caches vides), « init engine » 52,6 s contre 6,0 s à chaud sur le 0.8B. À chaud, la compilation (cache AOT) ne pèse plus que 1,6 à 4,5 s ; il reste environ 15 s entre la bannière et le chargement du modèle et 12 s entre la fin de l'initialisation du moteur et l'ouverture du serveur. Face à vLLM, l'émission d'Herbert (2 à 3 ms) et la construction par le pilote (1,5 s à froid) font gagner l'essentiel de son temps de compilation à froid ; face à llama.cpp, rien, il ne compile pas au démarrage.
