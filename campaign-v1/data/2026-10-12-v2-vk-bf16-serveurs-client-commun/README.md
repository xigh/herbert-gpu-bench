# 2026-10-12 — V2 : les quatre moteurs en serveur, le même client (4080, horloges verrouillées)

Décision après V1 (#541) : les bancs de chaque moteur ne mesurent pas la même chose (llama-bench : prefill seul sur tokens synthétiques ; vLLM : sa latence de bout en bout ; Herbert : son compteur interne). V2 sert les quatre moteurs en HTTP et les chronomètre avec **le même client** (`_scripts/_v2_http.py`) : API Anthropic Messages (`/v1/messages`), en flux, greedy, 128 tokens générés, une requête à la fois, sans cache de préfixe (serveurs lancés sans, cinq passages différents par longueur, 3 tours, un échauffement écarté). TTFT = premier delta non vide reçu − envoi ; décodage = (tokens générés − 1) / (dernier delta − premier delta), tokens générés et tokens de prompt tels que chaque serveur les annonce. Passages : `/opt3/bench-qwen35/prompts.json` (ceux de `http_bench.py`).

Serveurs : **herbert-vk-serve** (`herbert-vk-bf16` 9738be7, feature `serve` : l'API Messages portée du serveur de herbert-rs, sur le gabarit et le moteur de herbert-vk, une requête à la fois, cache vide à chaque requête) ; **vLLM 0.28.0** `serve` (`--no-enable-prefix-caching`, `--max-model-len 4608`) ; **llama-server** 08ffba21e CUDA et Vulkan (`-np 1 -c 4608 --no-cache-prompt`, `cache_prompt: false` dans chaque requête, réglages par défaut sinon). Pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz (`beast/env-start.txt`, `beast/clocks-end.txt`).

Même prompt pour tous : le gabarit officiel du **0.8B ferme la réflexion par défaut** (`<think>\n\n</think>\n\n`, 2 tokens de plus), celui du 4B l'ouvre ; Herbert et llama-server l'ouvrent pour les deux. Le premier passage de vLLM 0.8B (réflexion fermée) est gardé (`beast/http-vllm-0.8B.json`) ; celui du tableau est refait avec `--default-chat-template-kwargs '{"enable_thinking": true}'` (`beast/vllm-think/`) : les quatre serveurs annoncent alors les mêmes tokens de prompt à chaque longueur.

## Résultats (`v2.summary`, médianes de 15 requêtes ; TTFT en ms / décodage en tok/s)

| modèle | tokens de prompt | Herbert | vLLM | llama.cpp CUDA | llama.cpp Vulkan |
|---|---|---|---|---|---|
| 0.8B | 20–21 | 6,49 / 361,5 | 18,92 / 353,9 | 12,01 / 300,3 | 23,20 / 296,5 |
| 0.8B | 109–111 | 7,21 / 366,6 | 19,04 / 354,9 | 25,31 / 300,1 | 39,98 / 294,7 |
| 0.8B | 510–511 | 13,08 / 357,5 | 20,25 / 354,7 | 34,02 / 297,8 | 65,77 / 293,6 |
| 0.8B | 2 058–2 059 | 44,75 / 340,4 | 46,17 / 349,4 | 93,93 / 296,9 | 212,18 / 286,1 |
| 0.8B | 4 105–4 107 | 86,54 / 317,0 | 87,16 / 337,4 | 159,65 / 293,2 | 392,41 / 282,8 |
| 4B | 20–21 | 22,42 / 76,57 | 26,21 / 74,17 | 39,27 / 70,98 | 92,25 / 69,31 |
| 4B | 109–111 | 23,78 / 76,52 | 27,77 / 74,09 | 79,03 / 70,98 | 171,19 / 69,17 |
| 4B | 510–511 | 53,69 / 76,04 | 57,54 / 73,78 | 121,35 / 70,78 | 283,12 / 69,01 |
| 4B | 2 058–2 059 | 216,35 / 74,62 | 210,94 / 72,50 | 343,57 / 70,45 | 959,92 / 68,57 |
| 4B | 4 105–4 107 | 418,54 / 73,13 | 403,51 / 72,12 | 602,47 / 69,95 | 1 807,90 / 68,18 |

- Premier token : Herbert devant vLLM jusqu'à 510 tokens (0.8B : 34 à 65 % de son temps ; 4B : 86 à 93 %), à égalité sur le 0.8B au-delà (96,9 et 99,3 %), derrière sur le 4B à 2 058 et 4 105 tokens (102,6 et 103,7 % du temps de vLLM). Devant les deux llama-server partout (llama-server CUDA : 28 à 70 % de son temps).
- Décodage : Herbert devant vLLM sur le 4B à toutes les longueurs (101,4 à 103,3 %) et sur le 0.8B jusqu'à 510 tokens (100,8 à 103,3 %), derrière sur le 0.8B à 2 058 (97,4 %) et 4 105 tokens (94,0 %) : son décodage perd plus vite avec la profondeur. Devant llama-server partout (CUDA : 104,6 à 122,2 %).
- llama-server : TTFT très dispersé (jusqu'à 102 % d'écart entre requêtes) et prefill servi nettement sous ce que llama-bench mesure au même commit (0.8B, 4 105 tokens : ≈ 25 700 tok/s contre 32 782 en V1).
- Client contre compteur interne d'Herbert : 0.8B à 4 105 tokens, 86,5 ms vus du client contre 78,5 ms de prefill dans le log du serveur (`server-herbert-0.8B.log`) : tokenisation, HTTP et premier token compris.

Fichiers : `beast/` (journaux des serveurs, sorties et JSON du client, `vllm-think/`, `run.out`), `v2.summary`, logs de construction, `_scripts/`.
