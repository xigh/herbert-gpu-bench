# Mes prédictions pour la campagne définitive de #002, le 12 octobre 2026 à 9 h 06

Écrites avant le lancement de la campagne définitive (binaire publié, llama.cpp CUDA et Vulkan au même commit, coût fixe de vLLM mesuré, horloges de la 4080 verrouillées à 2 805 et 11 201 MHz le même matin). Elles viennent **après** la première série de mesures du 6 au 8 octobre (wiki #497 à #528), que je connais : l'article devra le dire.

## Ses mots, tels quels (dictés, « profil » = prefill)

« Ma prédiction, c'est que vLLM restera toujours devant en mode CUDA et qu'on va s'approcher pas mal de llama.cpp en mode CUDA, qu'on va les dépasser en Vulkan, c'est sûr. Ça, ça vaut pour le profil essentiellement. Pour le décode, je pense qu'on sera à peu près à même hauteur. Peut-être sauf pour vLLM. Parce qu'il y a des spécificités. Peut-être que je n'utilise pas encore très bien la bande passante de la VRAM. Il faudra faire les tests aussi sur Omnibook 1, bien sûr. »

## Ce qu'on vérifiera, une ligne par prédiction

- **P1, prefill** : vLLM reste devant Herbert.
- **P2, prefill** : Herbert s'approche de llama.cpp CUDA.
- **P3, prefill** : Herbert dépasse llama.cpp Vulkan (« c'est sûr »).
- **P4, décodage** : Herbert à peu près à la même hauteur que les autres moteurs, sauf peut-être vLLM, devant. Cause avancée : la bande passante de la VRAM peut-être pas encore bien utilisée.
- **P5** : la campagne passe aussi sur l'omnibook (Arc 140V).

## Pourquoi il les tient (ajouté le même matin, avant le lancement)

« Ma prédiction, je l'appuie parce que je sais d'avance, quand on verra le CUDA, qu'il y a des fonctionnalités qui sont disponibles en CUDA qui ne sont pas disponibles en Vulkan. Ce sera tout l'intérêt de la publication de l'article #003. »

La lecture de « devant », « s'approcher » et « à même hauteur » se fera longueur par longueur et modèle par modèle, sans retoucher ce texte.
