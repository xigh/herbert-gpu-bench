# 2026-10-12 — V1 : campagne définitive de #002 (Vulkan), binaire publié, llama.cpp au même commit, quatre machines

En préparant #002, après trois défauts trouvés dans la première série (journal, 07:00) : llama.cpp CUDA de beast était un build du 17 juin (`bae36efa3`, build 9679) et non `c841aeeb8` comme l'écrivent les README de #510 et #528 ; vLLM est chronométré de bout en bout, Herbert et llama.cpp sur le prefill seul ; les chiffres de l'Arc et de Cezanne venaient de f26595d et 70e3d5b. Mes prédictions, écrites à 07:06 UTC avant toute mesure de V1 (`predictions.md`), protocole (`protocol.md`), journal (`journal.md`). Copie de travail : `~/Sources/herbert-gpu-bench/campaign-v1` (local).

Versions : `herbert-vk-bf16` **2f3ae48** (bf16 par défaut, code de e692613 aux commentaires près ; mêmes ids que #540 sur les 20 sorties de la 4080, `beast-marges.txt`) ; llama.cpp **08ffba21e** (build 11564, 12 octobre 2026), CUDA 12.8 sm_89 et Vulkan compilés sur beast, Vulkan sur white2 et l'omnibook, archives officielles `b11564` sur g5500 (SHA-256 dans `logs` du dépôt de campagne) ; vLLM **0.28.0** (torch 2.13.0+cu130). Une requête à la fois. N = 5 tours alternés, médianes, écart (max − min) / médiane entre parenthèses dans les `*.summary`.

| machine | GPU | pilote | horloges |
|---|---|---|---|
| beast | RTX 4080 16 Go | NVIDIA 595.99.02 | SM 2 805 et mémoire 11 201 MHz verrouillées, relevées au début, au milieu et à la fin (`beast/env-*.txt`, `beast/floor/clocks-*`) |
| beast | iGPU Ryzen 9 9900X (RADV Raphael) | Mesa | non verrouillées |
| white2 | iGPU Ryzen 7 5700G (RADV Renoir) | Mesa 25.2.8, noyau 7.0.0-38 | non verrouillées |
| omnibook | Arc 140V (Core Ultra 7 258V) | Mesa 26.0.3 (ANV), noyau 7.0.0-31 | non verrouillées |
| g5500 | RTX 3060 Ti 8 Go, Windows 11 | NVIDIA 591.86 | `g5500/` sans verrouillage ; `g5500-locked/` après `-lgc 2100 -lmc 7001` : obtenu en charge 1 650–1 950 MHz (médiane 1 890) et mémoire 6 801 MHz en P2 tout du long, jusqu'à 83 °C (`g5500-locked/clocks.csv`) |

## 4080 (beast) : prefill selon la longueur (`beast.summary`)

Herbert en part de chaque moteur, 128 / 256 / 512 / 1 024 / 2 048 / 4 000 tokens :

- 0.8B : vLLM 288,9 / 219,8 / 129,0 / 113,0 / 109,9 / 108,3 % ; llama.cpp CUDA 126,2 / 152,9 / 148,8 / 166,8 / 177,7 / 173,0 % ; llama.cpp Vulkan 388 à 496 %.
- 4B : vLLM 108,6 / 110,8 / 98,9 / 105,5 / 100,1 / 98,7 % ; llama.cpp CUDA 109,1 / 112,5 / 111,3 / 123,2 / 123,7 / 123,7 % ; llama.cpp Vulkan 325 à 437 %.
- llama.cpp CUDA au commit du jour est plus rapide que le build de juin (0.8B 21 653 à 33 935 tok/s contre 18 752 à 30 726 en #528 ; 4B 5 745 à 8 836 contre 5 189 à 7 928), avec une forte dispersion aux prompts courts (41 % à 128 tokens en 0.8B).

Latence avec un prompt court et un token généré (`beast-floor.txt`, ms) : 0.8B Herbert 4,05 / 4,15 / 4,69 à 16 / 64 / 128 tokens, vLLM 11,23 / 13,22 / 13,53 (4,65 avec un seul token) ; 4B Herbert 19,26 / 18,01 / 20,42, vLLM 19,07 / 21,12 / 22,18 (15,96 avec un token). Herbert : tokens du prompt / son débit de prefill (le compteur en secondes de la ligne `[turn]` n'a que la milliseconde) ; vLLM : sa latence de bout en bout.

Réglages de llama.cpp (`q-*.json`, micro-lot 256 à 2 048 × flash attention 0/1, 3 tours, comparés dans le même balayage au défaut, micro-lot 512) : en CUDA, le défaut avec flash attention est le meilleur réglage du 4B, et le micro-lot de 1 024 gagne 4,7 et 6,0 % sur le 0.8B à 2 048 et 4 000 tokens (34 789 et 34 735 tok/s) ; en Vulkan, le micro-lot de 2 048 avec flash attention gagne 15,7–17,3 % sur le 0.8B (14 062 tok/s à 2 048 tokens) et 12,9–13,1 % sur le 4B (2 818). Au meilleur réglage, llama.cpp reste derrière Herbert sur la 4080 dans les deux backends.

## p3 et p4 (prefill puis 255 tokens générés)

| GPU | modèle | forme | Herbert prefill / décodage | vLLM | llama.cpp CUDA | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| 4080 | 0.8B | p3 | 53 939 / 335,8 (`host.poll=1` : 380,7) | 46 080 / 350,1 | 25 407 / 313,6 | 11 335 / 307,5 |
| 4080 | 4B | p3 | 10 789 / 75,5 (77,6) | 10 250 / 73,2 | 7 816 / 71,6 | 2 512 / 70,4 |
| 4080 | 0.8B | p4 | 57 740 / 322,3 (354,3) | 54 330 / 336,7 | 32 462 / 310,0 | 11 686 / 299,7 |
| 4080 | 4B | p4 | 10 728 / 73,8 (76,0) | 10 807 / 72,0 | 8 574 / 71,0 | 2 427 / 69,8 |
| 3060 Ti (verrouillée) | 0.8B | p3 | 20 033 / 227,7 | — | 9 988 / 174,7 | 9 015 / 174,7 |
| 3060 Ti (verrouillée) | 2B | p3 | 9 036 / 101,2 | — | 5 989 / 95,6 | 5 476 / 89,9 |
| 3060 Ti (verrouillée) | 0.8B | p4 | 18 799 / 214,3 | — | 11 210 / 171,9 | 9 472 / 171,4 |
| 3060 Ti (verrouillée) | 2B | p4 | 9 014 / 98,4 | — | 6 395 / 94,9 | 5 613 / 89,1 |
| Arc 140V | 0.8B | p3 | 2 043 / 60,9 | — | — | 1 426 / 54,3 |
| Arc 140V | 4B | p3 | 333 / 12,03 | — | — | 283 / 11,37 (dispersion 40 %) |
| Arc 140V | 0.8B | p4 | 1 355 / 58,2 | — | — | 1 205 / 52,9 |
| Arc 140V | 4B | p4 | 272,2 / 11,74 | — | — | 273,0 / 11,19 |
| Cezanne | 0.8B | p3 | 550 / 22,04 | — | — | 436 / 19,76 |
| Cezanne | 4B | p3 | **64,4** / 4,02 | — | — | **69,4** / 3,88 |
| Raphael | 0.8B | p3 | 183,2 / 29,6 | — | — | 166,0 / 25,2 |
| Raphael | 4B | p3 | **19,8** / 4,65 | — | — | **26,2** / 3,91 |

Herbert à chaud pour p3 (le run à froid est dans les summaries ; sur la 3060 Ti non verrouillée, son prefill varie de 82 % d'un tour à l'autre). vLLM : prefill = 759 ou 2 880 / L(1 token) ; décodage = 254 / (L(255) − L(1)) ; un seul passage de son banc (5 itérations).

Cezanne, 4B : #507 donnait 76,0 tok/s au commit 70e3d5b le 7 octobre. Le même 70e3d5b recompilé et alterné aujourd'hui avec 2f3ae48 (`white2-cezanne/ab/`, 3 tours) : 64,7 contre 64,3 ; `conv.cols=0` 64,7, `prefill.pipe=0` 64,3. Pas une régression d'Herbert : la machine a changé (noyau 7.0.0-38 installé le 6 octobre à 16:49, redémarrage le 8 ; le noyau du 7 n'est pas connu).

## Tokens contre l'oracle CPU

4080 : identiques à #540 (`beast-marges.txt`). 3060 Ti, oracle `herbert-bf16-chat` v0.1.0 compilé sous MSVC (C1) : 0.8B p1 identique, p2 token 91 (égalité), p3 token 20 (1 quantum), p4 token 83 (égalité) ; 2B p1 et p3 identiques jusqu'au bout, p2 token 15 et p4 token 46 (1 quantum) (`g5500-locked.marges`).

## Fichiers

`beast/` (env, `cmp-*`, `pp-*`, `p3/`, `p4/`, `q-*`, `p3-raphael/`, `floor/`, `run.out`), `white2-cezanne/` (dont `ab/`), `omnibook-arc/`, `g5500/` et `g5500-locked/` (dont `cmp-*`, `clocks.csv`), `*.summary`, `beast-floor.txt`, `*-marges*`, logs de construction (`build-*` ; le premier essai CUDA de beast, interrompu, n'a pas de log : sa sortie partait vers `/dev/null`), `protocol.md`, `predictions.md`, `journal.md`, outillage `_scripts/`.
