# 2026-10-12 — herbert-vk-bf16 : les marges au premier écart contre l'oracle CPU (4080, pilote 595.99.02)

En préparant #002 : le brief et le cadrage parlé disaient « tokens identiques à la référence CPU », or `dr2-beast/cmp-*` et `dr3-beast/cmp-*` de #527 montrent un premier écart sur la plupart des prompts, et le 4B à 512 tokens diverge dès le token 0. Cette passe relève, au premier écart, les logits GPU de son propre choix et du choix du CPU (`HERBERT_VK_TOP=5`), comme `top-p2/p3-nvidia.log` de #497.

4080 de beast, après le redémarrage du 12 octobre 2026 : le pilote 595.99.02 installé le 10 octobre par les mises à jour automatiques n'était chargé que côté utilisateur (noyau 595.91.07, NVML en échec, Vulkan ne voyait plus que l'iGPU RADV), il l'est maintenant aussi côté noyau (`top1-beast/env.txt`). Horloges non verrouillées : seuls les ids comptent ici. Binaire `target-bf16/release/herbert-vk-chat` du 8 octobre (celui de #528, défauts NVIDIA de e692613, `commits.txt`), sans `--tune`. Prompts et options de `compare.sh` (p3 avec réflexion et 256 tokens, les autres `--no-think` et 200 tokens, pp4000 avec `--ctx 4608` comme dr3). Logs CPU repris de `dr2-beast/cmp-*` (même oracle). Outillage : `_top1.sh`, `_marges.py`.

## Résultat (`top1.marges`)

Les premiers écarts sont exactement ceux de #527 sur le pilote précédent, prompt par prompt (0.8B : p2 15, p3 172, p4 8, pp128 82, pp256 38, pp512 80, pp1024 73, pp2048 48 ; 4B : p2 32, p3 85, p4 105, pp128 4, pp256 24, pp512 0, pp1024 11, pp2048 64, pp4000 42). Identiques sur toute la sortie : p1 (les deux tailles) et pp4000 du 0.8B.

À chaque premier écart, le choix du CPU est dans le top 5 du GPU, à 0, 1 ou 2 quanta bf16 du choix du GPU : 8 écarts à 0 (égalité exacte des logits bf16 côté GPU), 7 à 1 quantum, 1 à 2 quanta (0.8B p4, token 8 : 29,5 contre 29,25). Le 4B à 512 tokens, token 0 : 27775 à 24,625 contre 2064 (choix CPU) à 24,5, un quantum.

Les deux argmax émis prennent le premier indice en cas d'égalité (`herbert-jit-bf16` x86 `argmax_bf16` : « strict float `>`: first tie wins » ; `herbert-vk-bf16` `argmax_final` : « first index on a tie »). Aux 8 égalités côté GPU, le CPU a choisi un id plus grand : ses propres logits bf16 n'étaient donc pas à égalité. Les écarts viennent de l'arrondi des logits, pas d'une règle de départage différente.
