# 2026-10-12 — V3 : le bf16 est-il le chemin faible de llama.cpp ? BF16, F16 et Q8_0 sur la 4080

Ma question après V1 et V2 (#541–#543), que je trouvais « incroyables » : llama.cpp est-il pénalisé par le bf16, qui n'est pas son chemin principal ? llama-bench 08ffba21e, CUDA et Vulkan, le même modèle en BF16, F16 (mêmes octets) et Q8_0 (ce que les utilisateurs font tourner). 0.8B : GGUF de `/opt3/bench-qwen35` (convertis le 29 août) ; 4B : F16 et Q8_0 fabriqués ici depuis le GGUF BF16 avec `llama-quantize` 08ffba21e (`quantize-4B-*.log`, `gguf.sha256`). 4080 de beast, horloges verrouillées (2 805 / 11 201 MHz au début et à la fin), lancé de l'extérieur par `beast-ext`. Prefill de 128 à 4 000 tokens (`-r 5`), décodage de 255 tokens aux profondeurs 759 et 2 880 (`-r 5`). Herbert bf16 repris de V1 (#541). Tableaux : `v3.summary`.

## Ce que ça change

- **Vulkan, prefill : le BF16 de llama.cpp est 2,3 à 3,6 fois plus lent que son F16** (0.8B 11 876 contre 26 732 tok/s à 512 tokens ; 4B 2 484 contre 8 965). L'écart « 3,3 à 5 fois » d'Herbert sur llama.cpp Vulkan (#541) vient surtout de là : face au F16, Herbert est à 145–217 % (0.8B) et 108–121 % (4B).
- **CUDA, prefill : le F16 gagne jusqu'à 18–20 % sur le BF16.** Face à llama.cpp CUDA F16, Herbert est à 126–151 % sur le 0.8B, mais **93–110 % sur le 4B** : derrière à 512 tokens (9 688 contre 10 413), devant de 1,5 à 3,6 % au-delà de 1 024 tokens, de 10 % à 128.
- **Décodage : BF16 et F16 identiques** dans les deux backends (bande passante, mêmes octets) : les comparaisons de décodage de V1 et V2 tiennent telles quelles.
- **Q8_0** (autre contrat numérique, poids quantifiés) : décodage 1,35 fois (0.8B) à 1,6 fois (4B) celui d'Herbert bf16 (deux fois moins d'octets) ; prefill CUDA du 4B devant Herbert de 3,5 à 14,5 %. C'est le sujet de l'article sur la quantification.
- **À refaire avec ce constat** : les comparaisons au prefill contre llama.cpp sur les autres GPU (Arc, iGPU AMD, 3060 Ti) et au client commun (llama-server avait reçu le GGUF BF16) ; sur les iGPU AMD, où Herbert perdait déjà contre le BF16, le F16 creusera l'écart.
