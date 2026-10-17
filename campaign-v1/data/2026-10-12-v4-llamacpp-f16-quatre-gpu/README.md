# 2026-10-12 — V4 : llama.cpp en F16 (son chemin 16 bits fort) contre Herbert bf16, au banc et au client commun

Suite de #544 (le BF16 est le chemin faible de llama.cpp, surtout en Vulkan) : llama.cpp 08ffba21e en **F16** (GGUF fabriqués sur chaque machine depuis le GGUF BF16 avec `llama-quantize` 08ffba21e, empreintes `*/gguf-f16.sha256`, identiques d'une machine à l'autre pour le même modèle), contre les chiffres d'Herbert bf16 du même jour (banc : V1 #541 ; client : V2 #542, #543). Herbert n'a pas changé. Banc : llama-bench aux formes p3/p4 (prefill N puis 255 tokens à la profondeur N), 5 tours de `-r 1`. Client : llama-server F16 chronométré par `_v2_http.py` comme en V2. Tableaux : `v4.summary`.

Machines : beast (iGPU Raphael au banc, 4080 verrouillée au client, CUDA et Vulkan), omnibook (Arc 140V, banc p3 et p4, client), white2 (iGPU Cezanne, banc). **g5500 (3060 Ti) : non fait.** La machine a fait un écran bleu pendant le banc (`DPC_WATCHDOG_VIOLATION` 0x133, 15 h 36 heure locale, horloges verrouillées au maximum depuis le matin), relancé sans verrouillage, puis la machine est devenue injoignable (plus de ping) ; données partielles restées sur la machine.

## Ce que ça change

- **Intel et AMD, prefill : llama.cpp F16 devant Herbert.** Herbert à 47 % (4B) et 70 % (0.8B) de llama.cpp sur Raphael, 50 % et 72 % sur Cezanne, 56–58 % (4B) et 83–95 % (0.8B) sur l'Arc. Contre le BF16 (#541), Herbert était devant partout sauf le 4B des iGPU AMD.
- **Intel et AMD, décodage : Herbert reste devant**, de 4 à 18 %.
- **4080, client commun : Herbert reste devant llama-server F16.** Premier token en 29 à 80 % du temps du meilleur des deux llama-server (CUDA ou Vulkan) ; décodage +4,5 à +22 %. Le F16 rend llama-server Vulkan 1,6 à 3 fois plus rapide au premier token que son BF16 ; le CUDA gagne jusqu'à 13 %.
- **Arc, client commun : Herbert devant au premier token jusqu'à 500 tokens environ, derrière au-delà** (4B : 141 % et 159 % du temps de llama-server F16 à 2 058 et 4 105 tokens) ; décodage devant (+6 à +19 %).
