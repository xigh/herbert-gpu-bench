# 2026-10-06 — herbert-vk-bf16 : référence de la phase 2 (profil par noyau, balayage matvec, vitesse 0.8B et 4B)

Étalon des deux pistes de la phase 2 (prefill par blocs, decode par vendeur). Binaires compilés à 10:20 depuis l'arbre de travail **non commité** au-dessus de f6ce76f (`commit.txt`, `working-tree-diffstat.txt`) : subgroup porté par chaque pipeline, `Tuning` (`--tune`), `--profile` (horodatage GPU par opération), matvec à `matvec.rows` lignes par subgroup, sonde `--sweep`, propriétés NV2. Chemin par défaut du decode identique à celui de #497 (1 ligne par subgroup, mêmes noyaux). Machines au repos (charge < 0,2 au départ, `uptime` en tête de chaque tour) ; beast exclu (une autre campagne en cours).

| étape | fichiers | résultat |
|---|---|---|
| sonde + balayage matvec | `probe-sweep-white2.log`, `probe-sweep-omnibook.log` | NV2 sur la 4080 : workgroupScope, flexibleDimensions, tensorAddressing… ; 45 configurations flexibles dont bf16→f32 en portée workgroup 32/32/16 (256 invocations), max 1 024, 8 192 B réservés. Matvec (wave × 1/2/4 lignes, 5 tours alternés) : 590–690 GB/s (4080), 95–108 (Arc), 30–34 (Cezanne), écarts entre variantes ≤ 3 % ; `lm_head` 4B sur la 4080 à 310 GB/s quelle que soit la variante |
| profils decode 0.8B | `profile0-{nvidia,amd,intel}-Qwen3.5-0.8B.log` | p3, 1 014 pas, médianes : pas GPU 3 984 µs (4080), 18 918 (Arc), 51 691 (Cezanne) ; 4080 : projections ~61 %, `dn_recurrence` 16,8 %, `sdpa_decode` 13,2 % ; Arc : ~76 %, 8,1 %, 8,1 % ; Cezanne : ~88 %, 4,7 %, 5,3 % |
| vitesse | `speed2-white2/`, `speed2-omnibook/`, `_speed.sh`, `_speed.py` | p3 (759 + 255 tokens), N = 5 alternés avec le CPU, médianes decode : 0.8B 4080 224,57 · Cezanne 18,72 · CPU white2 19,29 · Arc 49,90 · CPU omnibook 37,02 ; 4B 4080 64,93 · Cezanne 3,62 · CPU white2 3,63 · Arc 11,36 · CPU omnibook 6,83 tok/s (prefill Vulkan token par token) |
| validation | `spirv-val-3.log` | 57 modules valides |
