# 2026-10-12 — V2 sur l'Arc 140V et la RTX 3060 Ti : Herbert contre llama-server, le même client

Suite de #542 (même protocole, même définition du client) à ma demande : `herbert-vk-serve` (9738be7) contre llama-server 08ffba21e (omnibook, Vulkan compilé sur place) ou `b11564` (g5500, archives officielles CUDA 12.4 et Vulkan). API Messages en flux, greedy, 128 tokens, sans cache de préfixe, cinq passages × 3 tours par longueur, mêmes tokens de prompt annoncés par tous les serveurs.

- **omnibook** (Arc 140V, Mesa 26.0.3) : client Python `_v2_http.py`, 0.8B et 4B.
- **g5500** (RTX 3060 Ti 8 Go, Windows 11, pilote 591.86) : pas de Python sur la machine, client **`v2-client`** (`_scripts/_client`, Rust, la même logique ligne pour ligne). Contrôle sur beast contre le même serveur Herbert 0.8B (`client-check-beast/`, Python, Rust, Python) : premier token 0,5 à 0,7 ms plus tôt vu du client Rust (le coût du client Python), décodage identique à la dispersion près. 0.8B et 2B (le 4B ne tient pas).
- **Horloges de g5500** (`g5500/clocks.csv`, verrouillage demandé à 2 100 / 7 001 MHz, en charge 1 410–1 905 MHz) : la mémoire dépend du moteur. llama-server CUDA en P2 (6 801 MHz) tout du long, llama-server Vulkan en P0 (7 001), Herbert en P2 pour le 0.8B et en P0 pour le 2B. 2,9 % de fréquence mémoire d'écart : à dire avec les chiffres de décodage du 2B.

## Résultats (`v2b.summary` ; TTFT en ms / décodage en tok/s, médianes de 15 requêtes)

| GPU | modèle | tokens de prompt | Herbert | llama-server CUDA | llama-server Vulkan |
|---|---|---|---|---|---|
| Arc 140V | 0.8B | 20–21 | 38,95 / 62,58 | — | 162,49 / 53,84 |
| Arc 140V | 0.8B | 4 105–4 107 | 3 675 / 56,63 | — | 4 304 / 50,71 |
| Arc 140V | 4B | 20–21 | 127,9 / 12,17 | — | 580,9 / 11,38 |
| Arc 140V | 4B | 510–511 | 1 502 / 12,11 | — | 2 301 / 11,33 |
| Arc 140V | 4B | 4 105–4 107 | 17 160 / 11,64 | — | 17 598 / 11,08 |
| 3060 Ti | 0.8B | 20–21 | 10,26 / 234,81 | 30,28 / 166,14 | 25,77 / 189,50 |
| 3060 Ti | 0.8B | 4 105–4 107 | 246,7 / 207,61 | 441,8 / 161,95 | 523,8 / 181,32 |
| 3060 Ti | 2B | 20–21 | 17,01 / 104,87 | 36,02 / 93,33 | 38,83 / 94,71 |
| 3060 Ti | 2B | 4 105–4 107 | 536,5 / 99,10 | 748,2 / 92,10 | 871,1 / 92,62 |

Herbert devant à toutes les longueurs. Premier token : Arc 21 à 85 % du temps de llama-server (0.8B), 22 à 97,5 % (4B, quasi égalité à 4 106 tokens) ; 3060 Ti 25 à 56 % de llama-server CUDA (0.8B), 37 à 72 % (2B). Décodage : Arc +12 à +17 % (0.8B), +5 à +7 % (4B) ; 3060 Ti +28 à +42 % sur llama-server CUDA (0.8B), +8 à +12 % (2B). Servi, llama.cpp décode plus vite en Vulkan qu'en CUDA sur la 3060 Ti (189 contre 166 tok/s en 0.8B), en P0 contre P2. Le décodage d'Herbert baisse avec la longueur du contexte plus vite que celui de llama-server (0.8B 3060 Ti : 234,8 → 207,6 contre 166,1 → 162,0), comme face à vLLM sur la 4080.

Fichiers : `omnibook-arc/`, `g5500/` (journaux des serveurs, sorties et JSON du client, `clocks.csv`, logs de construction), `client-check-beast/`, `v2b.summary`, `_scripts/` (pilotes, client Python, client Rust).
