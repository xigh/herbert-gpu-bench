# 2026-10-13 — Le chargeur de herbert-vk : fichiers mappés, cache de poids f16, conversion et envoi sur tous les cœurs ; le démarrage d'Herbert refait (V7b)

Suite de V7 (#551) : le démarrage d'Herbert était dominé par le chargement des poids (4B : 4,1 s en bf16, 18,7 s en f16), pas par l'émission du SPIR-V (2 à 3 ms). La version CPU a un cache de poids : le porter, et accélérer le chargement.

**Le chargeur** (herbert-vk-f16 2ac926d ; reporté sans le cache dans herbert-vk-bf16 8440202, `commits.txt`). Avant : le checkpoint entier copié élément par élément dans des vecteurs hôtes, recopié pour les poids en plusieurs morceaux, puis copié vers la carte, sur un cœur ; en f16, conversion élément par élément sous un verrou. Après : les fichiers mappés, un tenseur bf16 prêté tel quel depuis la projection du fichier, les poids en plusieurs morceaux (gate+up, projection DeltaNet, q+k+v) écrits à leur place dans un seul tampon de la carte, l'envoi vers la mémoire de la carte (ReBAR sur la 4080) réparti sur les cœurs. Version f16 : conversion sur tous les cœurs, une fois, dans un cache `~/.cache/herbert/<modèle>/vk-f16_v1.herbcache` (la racine des autres caches d'Herbert, même identité du modèle), mappé aux chargements suivants ; `HERBERT_CACHE=0` le contourne. Pas de plafond de taille (le `HERBERT_CACHE_MAX_GB` de la version CPU n'est pas porté).

**Mêmes poids au bit près** (`load-test/`, `load-test-bf16/`) : p1–p4, 0.8B et 4B, les 5 meilleurs logits et le texte identiques entre l'ancien et le nouveau chargeur, en f16 et en bf16 (herbert-vk-f16 e4901c4 contre 2ac926d ; herbert-vk-bf16 2f3ae48 contre 8440202).

| poids chargés, 4080 de beast | avant | après |
|---|---|---|
| 4B f16, cache présent | 18,3 s | 0,7 s |
| 4B f16, premier lancement (écrit le cache, 7,83 Gio) | 18,3 s | 3,8 s |
| 4B f16, sans cache (24 threads) | 18,3 s | 1,6 s |
| 4B bf16 | 3,9–4,0 s | 0,7 s |
| 0.8B f16, cache présent (1,40 Gio) | 3,2–3,3 s | 0,3 s |
| 0.8B bf16 | 0,7–0,8 s | 0,2–0,3 s |

**V7b, le démarrage refait** (`v7b/`, `v7b.summary`, `_scripts/_v7-startup.sh` avec `ENGINES`) : Herbert (2ac926d, bf16 et f16, cache f16 présent) et llama.cpp Vulkan dans les mêmes tours, à froid et à chaud, 5 tours. Premier token, en s :

| | 0.8B à froid | 0.8B à chaud | 4B à froid | 4B à chaud |
|---|---|---|---|---|
| Herbert bf16 | 2,11 | 0,62 | 2,54 | 1,04 |
| llama.cpp Vulkan BF16 | 3,09 | 0,78 | 4,95 | 1,47 |
| Herbert f16 | 1,85 | 0,64 | 2,32 | 1,10 |
| llama.cpp Vulkan F16 | 3,10 | 0,77 | 4,84 | 1,43 |

Herbert démarre maintenant avant llama.cpp Vulkan dans tous les cas. Contre les chiffres de V7 pour llama.cpp CUDA (même machine, autre série : 0,82 s et 1,26 s, sans différence froid / chaud), Herbert est devant à chaud (0,62 et 1,04 s), derrière à froid (2,1 et 2,5 s) : le pilote construit ses pipelines en 1,2 à 1,5 s quand son cache est vide, 4 ms quand il est plein ; l'émission reste à 2 à 3,5 ms.
