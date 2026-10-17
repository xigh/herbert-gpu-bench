# Journal de la campagne V1 (#002, Vulkan)

Heures en UTC.

**06:45–07:00, avant la campagne.** En relisant le cadrage de #002 contre les dépôts du wiki : les tokens du GPU ne sont pas « identiques à la référence CPU » (premiers écarts du token 0 au token 172, wiki #527) ; vérifié avec les logits (`HERBERT_VK_TOP=5`, wiki #540) : à chaque premier écart, les deux candidats sont à 0, 1 ou 2 quanta bf16. Beast avait reçu le pilote 595.99.02 le 10 octobre (mises à jour automatiques) sans redémarrer : Vulkan ne voyait plus la 4080. Redémarrage de beast vers 06:50.

**07:00, trois défauts du protocole de la première série.** (1) llama.cpp CUDA sur beast était un build du 17 juin (`bae36efa3`, build 9679), pas le `c841aeeb8` du Vulkan, contrairement à ce qu'écrivent les README de #510 et #528. (2) vLLM est chronométré de bout en bout par son banc, Herbert et llama.cpp sur le prefill seul : sur le 0.8B, la requête de vLLM ne descend pas sous ≈ 13,5 ms de 128 à 512 tokens. (3) Les chiffres de l'Arc et de Cezanne viennent de f26595d et 70e3d5b, pas du code à publier.

**07:06, mes prédictions** (`predictions.md`), avant toute mesure de V1, après la première série que je connais. J'ai verrouillé les horloges de la 4080 (2 805 et 11 201 MHz, deux commandes : `nvidia-smi` refuse les deux en une).

**07:06–07:16, préparation.** Décision : le build bf16 devient le défaut (`herbert-vk-bf16` 2f3ae48 : `default = ["bf16"]`, README, deux noms internes retirés des commentaires). llama.cpp pris au commit du jour, `08ffba21e` (build 11564), pour CUDA et Vulkan : compilé sur beast (le premier essai CUDA a échoué, cmake avait pris le `nvcc` 12.0 d'Ubuntu, refusé par gcc 13 ; refait avec `/usr/local/cuda/bin/nvcc` 12.8), sur white2 et l'omnibook (Vulkan) ; archives officielles `b11564` sur g5500 (SHA-256 dans `g5500-llama-dl.out`). Dans ce build, la 4080 de beast est `Vulkan1` (le 8 octobre, `Vulkan0`). g5500 : RTX 3060 Ti 8 Go, le 4B n'y tient pas, 0.8B et 2B.

**07:17, lancement sur beast** (`_v1-beast.sh`) ; 07:19 g5500 (`_v1-win.sh`, Git Bash). white2 et l'omnibook : un premier lancement à 07:18 a pris la mauvaise ligne de `--list-devices` (périphérique vide), arrêté et relancé à 07:21 après correction de `_v1-vk.sh`.

**07:35, tokens du binaire publié.** Les 20 sorties de la phase T sur la 4080 ont les mêmes ids que la passe #540 (binaire du 8 octobre, e692613 compilé avec `--features bf16`) : passer le bf16 en défaut n'a rien changé.
