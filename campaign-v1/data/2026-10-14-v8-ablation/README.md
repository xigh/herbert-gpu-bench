# 2026-10-14 — V8 : l'ablation d'Herbert sur la 4080 (fusions, soumission, enchaînement, taille des blocs)

Question soulevée à la relecture de #002 : séparer ce qu'apportent la fusion et la soumission groupée. En vérifiant, constat : `fuse.norm` et `fuse.swiglu` ne s'appliquent qu'au décodage (`src/vk/tuning.rs`), alors que #002 les invoquait pour le prefill ; texte corrigé. Les deux fusions font à elles seules l'écart 192 / 265 opérations du pas du 0.8B (241 sans `fuse.norm`, 216 sans `fuse.swiglu`, 265 sans les deux ; 4B : 254 / 351).

Mesure (`_scripts/_v8-ablation.sh`, `_v8_summary.py`) : herbert-vk-bf16 **8440202** (`~/herbert-vk-bf16-art` sur beast), compteur d'Herbert comme V1 (`--warmup 1`, tour mesuré ensuite, un processus par mesure), prompts de V1 (p1-short 24 tokens, pp512 513, pp2048 2 048, pp4000 4 001 : 1 token généré ; p3-long 761 + 256 générés pour le décodage), `--no-system --no-think --ctx 4608`. Variantes par `--tune` : `base` ; `nofuse` (`fuse.norm=0 fuse.swiglu=0`) ; `split1` (`prefill.split=1`, une soumission par couche au lieu du bloc entier) ; `nopipe` (`prefill.pipe=0`) ; `block256` (`prefill.block=256` au lieu de 2 048 / 1 024) ; `alloff` (tout). 5 tours, variantes alternées dans chaque tour, 0 erreur. Le prefill en ms est recalculé depuis le débit imprimé (les secondes n'ont que 3 décimales). Condition : une autre tâche en pause (SIGSTOP, 6,4 Go résidents, carte en P2, mémoire à 10 801 MHz au lieu de 11 201) ; contrôle avant lancement : 0.8B pp512 48 977 tok/s contre 49 210 en V1. Tableaux : `v8.summary`.

| temps de prefill / base | 0.8B 24 | 0.8B 513 | 0.8B 2 048 | 0.8B 4 001 | 4B 24 | 4B 513 | 4B 2 048 | 4B 4 001 |
|---|---|---|---|---|---|---|---|---|
| nofuse | ×1,00 | ×1,00 | ×1,01 | ×1,00 | ×0,98 | ×1,02 | ×1,00 | ×1,00 |
| split1 | ×1,05 | ×1,02 | ×1,01 | ×1,01 | ×1,01 | ×1,03 | ×1,00 | ×1,00 |
| nopipe | ×1,03 | ×1,02 | ×1,00 | ×1,01 | ×1,00 | ×1,02 | ×1,01 | ×1,01 |
| block256 | ×1,00 | ×1,44 | ×1,36 | ×1,49 | ×0,99 | ×1,41 | ×1,19 | ×1,20 |
| alloff | ×1,15 | ×1,63 | ×1,51 | ×1,64 | ×1,02 | ×1,44 | ×1,23 | ×1,25 |

Décodage (p3-long) : sans les fusions, 333,2 → 326,7 tok/s (0.8B, −2,0 %) et 72,8 → 71,5 (4B, −1,8 %) ; les autres réglages ne le touchent pas (±0,4 %).

Lecture : les grands blocs sont le levier principal du prefill (moins de passes, donc à la fois moins de lancements et moins de relectures des poids : V8 ne sépare pas les deux). Soumission en un bloc et enchaînement sans aller-retour : 1 à 5 %, surtout sur les prompts courts. Fusions : décodage seulement, 2 %. Tout désactivé, le prefill du 0.8B tombe à 30,0k (513), 38,8k (2 048) et 35,0k tok/s (4 001), de l'ordre de llama.cpp CUDA bf16 en V1 (33 078, 33 519, 32 782 ; autre série, autre horloge mémoire). Sur 24 tokens, ces réglages ne font que 15 % du temps du 0.8B, 2 % du 4B : le reste tient aux noyaux eux-mêmes (attention et récurrence sur les cœurs tensoriels, ordre des tuiles, wiki #504–#513), que V8 ne désactive pas.
