# Les données brutes de #002

Chaque dossier est copié tel quel depuis mes notes de travail, avec ses scripts (préfixés `_`), ses logs et ses résumés. Les numéros « #5xx » qu'on y croise renvoient à ces notes, qui restent privées : chaque dossier se lit sans elles. Heures en UTC.

## La première série, avant les prédictions (6 au 8 octobre 2026)

- `2026-10-06-vk-bf16-reference-phase2/` : le produit matrice-vecteur du décodage sur la RTX 4080, de 590 à 690 Go/s selon la variante.
- `2026-10-06-vk-bf16-phase2-variantes/` : la sonde en lecture pure de la même carte, 694 Go/s (`probe-gemm-4080.log`).
- `2026-10-07-vk-bf16-prefill-longueurs/` : le prefill selon la longueur ; le 4B à 128 tokens, le « point faible » du journal.
- `2026-10-08-vk-bf16-horloge-memoire-tuiles/` : le régime de l'horloge mémoire (`short1-beast/`, `lmc-*.log`), les tuiles plus grandes et les tuiles de 16 × 16 × 16 (`dr1-beast/`). Publié sans son résumé, qui anticipe le prochain numéro.

## La campagne V1 et ses suites (12 au 14 octobre 2026)

- `2026-10-12-v1-vk-bf16-campagne-definitive/` : V1, bf16 contre bf16, au banc de chaque moteur, sur les cinq GPU.
- `2026-10-12-v2-vk-bf16-serveurs-client-commun/` et `2026-10-12-v2-vk-bf16-client-commun-arc-3060ti/` : V2, le client commun (premier token et décodage vus de l'utilisateur), sur la 4080 puis sur l'Arc et la 3060 Ti.
- `2026-10-12-v3-llamacpp-bf16-f16-q8/` : V3, llama.cpp en bf16, f16 et q8 : le bf16 est son chemin faible.
- `2026-10-12-v4-llamacpp-f16-quatre-gpu/` : V4, Herbert bf16 face à llama.cpp f16 sur quatre GPU (la comparaison à contrats différents dont parle le journal).
- `2026-10-12-herbert-vk-f16-etape1/` et `2026-10-12-herbert-vk-f16-etape3-gemm-h16-amd/` : la construction de la version f16 d'Herbert, contrôlée contre la version bf16, et ses multiplications de matrices en f16 sur les GPU intégrés AMD.
- `2026-10-12-g5500-tdr-pilote-616/` : la RTX 3060 Ti qui décroche sous charge, avec deux pilotes.
- `2026-10-12-vk-bf16-marges-divergences/` : le contrat numérique ; chaque premier écart entre GPU et processeur, avec les cinq meilleurs candidats de chaque côté.
- `2026-10-13-v5-f16-contre-f16/` : V5, f16 contre f16 sur quatre GPU, au banc et au client commun.
- `2026-10-13-v6b-llamacpp-f16-meilleurs-reglages/` : V6b, le balayage des réglages de llama.cpp f16 (micro-lots, flash attention), un processus par réglage, Herbert dans les mêmes tours.
- `2026-10-13-v7-demarrage/` : V7, le démarrage jusqu'au premier token, à froid et à chaud, avec vLLM.
- `2026-10-13-chargeur-herbert-vk/` : le nouveau chargeur d'Herbert, mêmes poids au bit près, et le démarrage refait (V7b).
- `2026-10-13-spv-emis-deux-gpu/` : le SPIR-V émis pour la 4080 et pour le GPU intégré du Ryzen 9 9900X, désassemblé, et les seules lignes qui changent.
- `2026-10-14-v7c-demarrage-memes-tours/` : V7c, le démarrage d'Herbert, de llama.cpp Vulkan et de llama.cpp CUDA dans les mêmes tours : le tableau de l'article.
- `2026-10-14-v8-ablation/` : V8, l'ablation d'Herbert sur la 4080 : le même binaire, ses fusions, sa soumission en un bloc, l'enchaînement des blocs et la taille des blocs désactivés un par un.
