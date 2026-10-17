# 2026-10-12 — herbert-vk-f16, étape 3 : le GEMM SIMT en f16 « packed » sur les iGPU AMD — prefill +33 à +82 %, tokens tenus

Suite de #546 et #545 (llama.cpp F16 devant Herbert au prefill sur les iGPU AMD, qui calculent en f16 deux fois plus vite qu'en f32). `gemm_simt_h16` (herbert-vk-f16 e71fb42, `commits.txt`) : les tuiles de X et W rangées en paires f16 le long de K dans la mémoire partagée, chacune des 4 × 4 sorties d'une invocation accumulée dans un `f16vec2` (une FMA pour deux pas de K, `v_pk_fma_f16` sur AMD), reversée dans son accumulateur f32 toutes les `gemm.h16` tuiles de 16 (h1 : 8 produits par moitié en f16 avant le f32 ; h64 : presque tout en f16). Variante séparée, le GEMM f32 inchangé ; `gemm.h16` refusé hors feature `f16` ou sans `shaderFloat16`. `shaderFloat16` activé dès que le GPU le propose.

Banc (`_scripts/_h16-ab.sh`) : forme p3 (759 + 255), à chaud, cinq variantes alternées (f32, h1, h2, h4, h64), 3 tours, même build f16 ; puis p1–p4 contre l'oracle CPU avec les 5 meilleurs logits (logs CPU de #546 repris). iGPU Cezanne (white2, Mesa 25.2.8) et Raphael (beast). Tableaux : `h16.summary`.

| iGPU | modèle | prefill f32 → h1 (tok/s) | gain | h64 | llama.cpp Vulkan F16 (#545) |
|---|---|---|---|---|---|
| Cezanne | 0.8B | 544,4 → 732,4 | +34,5 % | +38,1 % | 761,1 |
| Cezanne | 4B | 62,9 → 110,2 | +75,2 % | +82,4 % | 129,9 |
| Raphael | 0.8B | 181,4 → 272,4 | +50,2 % | +51,3 % | 262,7 |
| Raphael | 4B | 19,7 → 26,1 | +32,5 % | +33,0 % | 42,1 |

Décodage inchangé (bande passante). Tokens : le premier écart avec l'oracle CPU tombe au même token qu'avec le GEMM f32 dans presque tous les cas, à quelques quanta f16 ; 4B p2 : identique jusqu'au bout en f32, écart au token 23 (0,0156) avec h1 sur Raphael et h4/h64 sur Cezanne ; h64 décale 4B p3 (151 → 117) et rend 0.8B p4 identique sur Cezanne. La période de reversement change la vitesse de moins de 3 % : h1 suffit.

Face à llama.cpp F16 (#545, même machine, même jour) : Cezanne 0.8B 96 %, 4B 85 % ; Raphael 0.8B 104 % (devant), 4B 62 %.
