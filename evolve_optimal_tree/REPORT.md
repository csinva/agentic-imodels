# pygosdt against the reference GOSDT: benchmark report

**63** (dataset, λ) pairs where both implementations returned a tree, out of 75: **57** identical objectives, **6** strictly better for pygosdt, **0** worse. pygosdt is **13.2× faster** (geometric mean of reference time ÷ pygosdt time; median 9.0×).

Setup: same CSV for both, single thread, sequential on an idle Apple M5 (16 GB), 600 s time cap and 6 GB memory cap for both, objectives recomputed independently from the returned trees, times exclude parsing and binarization. λ ∈ {0.1, 0.05, 0.02, 0.01, 0.005}. The interactive version of this report is `benchmarks/results/report.html`; the plot is `benchmarks/results/benchmark.png`.

## Fit: objective per (dataset, λ)

| dataset | λ=0.1 | λ=0.05 | λ=0.02 | λ=0.01 | λ=0.005 |
|---|---|---|---|---|---|
| chudi | same 0.4649 | same 0.2500 | same 0.1000 | same 0.0500 | same 0.0250 |
| monk_3 | same 0.5213 | same 0.3713 | same 0.2430 | same 0.1556 | same 0.0946 |
| monk_1 | same 0.4661 | same 0.3387 | same 0.1600 | same 0.0800 | same 0.0400 |
| iris | same 0.3400 | same 0.1900 | same 0.1000 | same 0.0600 | **py better** 0.0383 vs 0.0450 |
| monk_2 | same 0.4787 | same 0.4287 | same 0.3589 | same 0.2651 | same 0.1528 |
| tic-tac-toe | same 0.4466 | same 0.3966 | **py better** 0.3183 vs 0.3246 | same 0.2508 | no C++ tree (memory cap); py 0.1543 |
| gaussian_1k | same 0.3970 | same 0.2750 | **py better** 0.1850 vs 0.3170 | **py better** 0.1550 vs 0.3070 | **py better** 0.1400 vs 0.3020 |
| fico_1k | same 0.4910 | same 0.3910 | no C++ tree (memory cap); py 0.3310 | same 0.3110 | no C++ tree (time cap); py 0.3010 |
| sine_1k | same 0.5950 | no C++ tree (memory cap); py 0.5450 | same 0.4940 | same 0.4740 | **py better** 0.4000 vs 0.4620 |
| car_evaluation | same 0.3998 | same 0.2947 | same 0.2047 | same 0.1452 | same 0.1063 |
| coupon_bar | same 0.5119 | same 0.4539 | same 0.3716 | same 0.3411 | same 0.3186 |
| compas_binned | same 0.5611 | same 0.4611 | same 0.4011 | same 0.3749 | same 0.3560 |
| sine_10k | no C++ tree (memory cap); py 0.5996 | no C++ tree (memory cap); py 0.5496 | no C++ tree (memory cap); py 0.5181 | no C++ tree (memory cap); py 0.4981 | no C++ tree (memory cap); py 0.4881 |
| fico_binary | same 0.5040 | same 0.4040 | same 0.3440 | same 0.3240 | same 0.3087 |
| compas_processed | same 0.4114 | same 0.3614 | no C++ tree (memory cap); py 0.3279 | no C++ tree (memory cap); py 0.3079 | no C++ tree (memory cap); py 0.2979 |

### The six pairs where the trees differ

| dataset | λ | C++ objective | C++ errors/leaves | C++ status | pygosdt objective | pygosdt errors/leaves | pygosdt status |
|---|---|---|---|---|---|---|---|
| gaussian_1k | 0.02 | 0.3170 | 297/1 | time cap (10.0 min) | 0.1850 | 125/3 | optimal (18.9 s) |
| gaussian_1k | 0.01 | 0.3070 | 297/1 | time cap (10.1 min) | 0.1550 | 125/3 | optimal (1.4 min) |
| gaussian_1k | 0.005 | 0.3020 | 297/1 | time cap (10.8 min) | 0.1400 | 125/3 | optimal (4.5 min) |
| iris | 0.005 | 0.0450 | 3/5 | time cap (10.0 min) | 0.0383 | 2/5 | optimal (585 ms) |
| sine_1k | 0.005 | 0.4620 | 447/3 | time cap (12.5 min) | 0.4000 | 340/12 | time cap (10.0 min) |
| tic-tac-toe | 0.02 | 0.3246 | 196/6 | optimal (19.6 s) | 0.3183 | 190/6 | optimal (1.7 s) |

tic-tac-toe at λ=0.02 is a genuine reference error: it claims optimality at 0.3246 while a 6-leaf tree with objective 0.3183 exists. The other five are reference timeouts returning an incumbent (on sine_1k both timed out, pygosdt's incumbent ahead).

## Speed per dataset

| dataset | rows | source feats | binary feats | pairs with both trees | identical | py better | py worse | C++ optimal | py optimal | C++ total time | py total time | C++ ÷ py time (geo. mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chudi | 77 | 2 | 46 | 5 | 5 | 0 | 0 | 5 | 5 | 3.8 s | 40 ms | 23.6× |
| monk_3 | 122 | 11 | 11 | 5 | 5 | 0 | 0 | 5 | 5 | 1.3 s | 173 ms | 3.8× |
| monk_1 | 124 | 11 | 11 | 5 | 5 | 0 | 0 | 5 | 5 | 1.1 s | 31 ms | 10.0× |
| iris | 150 | 4 | 118 | 5 | 4 | 1 | 0 | 4 | 5 | 12.8 min | 751 ms | 632.8× |
| monk_2 | 169 | 11 | 11 | 5 | 5 | 0 | 0 | 5 | 5 | 2.4 s | 391 ms | 2.8× |
| tic-tac-toe | 958 | 9 | 27 | 4 | 3 | 1 | 0 | 4 | 5 | 4.4 min | 38.4 s | 7.7× |
| gaussian_1k | 1,000 | 1 | 999 | 5 | 2 | 3 | 0 | 2 | 5 | 32.9 min | 6.3 min | 16.6× |
| fico_1k | 1,000 | 23 | 1,357 | 3 | 3 | 0 | 0 | 0 | 1 | 43.2 min | 40.2 min | 5.6× |
| sine_1k | 1,000 | 1 | 999 | 4 | 3 | 1 | 0 | 0 | 1 | 57.3 min | 40.0 min | 5.3× |
| car_evaluation | 1,728 | 15 | 15 | 5 | 5 | 0 | 0 | 5 | 5 | 9.4 s | 841 ms | 6.6× |
| coupon_bar | 1,913 | 14 | 14 | 5 | 5 | 0 | 0 | 5 | 5 | 2.3 s | 221 ms | 6.9× |
| compas_binned | 6,907 | 12 | 12 | 5 | 5 | 0 | 0 | 5 | 5 | 72 ms | 14 ms | 3.9× |
| sine_10k | 10,000 | 1 | 9,999 | 0 | 0 | 0 | 0 | 0 | 0 | <1 ms | 50.0 min | – |
| fico_binary | 10,459 | 17 | 17 | 5 | 5 | 0 | 0 | 5 | 5 | 35.1 s | 3.2 s | 6.5× |
| compas_processed | 12,381 | 22 | 621 | 2 | 2 | 0 | 0 | 1 | 1 | 4.9 min | 30.2 min | 21,310.3× |

pygosdt was faster on 50 pairs; the 12 pairs where the reference was faster are trivial cases it finishes in 0–2 ms.

## Stop reasons (reference / pygosdt)

| dataset | λ=0.1 | λ=0.05 | λ=0.02 | λ=0.01 | λ=0.005 |
|---|---|---|---|---|---|
| chudi | optimal 2 ms / optimal 2 ms | optimal 43 ms / optimal 6 ms | optimal 567 ms / optimal 10 ms | optimal 1.3 s / optimal 11 ms | optimal 1.8 s / optimal 11 ms |
| monk_3 | optimal <1 ms / optimal <1 ms | optimal 5 ms / optimal 3 ms | optimal 53 ms / optimal 22 ms | optimal 264 ms / optimal 54 ms | optimal 1.0 s / optimal 94 ms |
| monk_1 | optimal <1 ms / optimal <1 ms | optimal 2 ms / optimal 3 ms | optimal 53 ms / optimal 8 ms | optimal 246 ms / optimal 9 ms | optimal 771 ms / optimal 10 ms |
| iris | optimal 128 ms / optimal 1 ms | optimal 2.1 s / optimal 2 ms | optimal 16.4 s / optimal 26 ms | optimal 2.5 min / optimal 138 ms | time cap 10.0 min / optimal 585 ms |
| monk_2 | optimal <1 ms / optimal <1 ms | optimal 5 ms / optimal 4 ms | optimal 75 ms / optimal 47 ms | optimal 444 ms / optimal 129 ms | optimal 1.9 s / optimal 211 ms |
| tic-tac-toe | optimal 2 ms / optimal 1 ms | optimal 163 ms / optimal 21 ms | optimal 19.6 s / optimal 1.7 s | optimal 4.1 min / optimal 11.5 s | memory cap 22.9 s / optimal 25.2 s |
| gaussian_1k | optimal 152 ms / optimal 10 ms | optimal 2.0 min / optimal 759 ms | time cap 10.0 min / optimal 18.9 s | time cap 10.1 min / optimal 1.4 min | time cap 10.8 min / optimal 4.5 min |
| fico_1k | time cap 13.6 min / optimal 9.5 s | time cap 11.2 min / time cap 10.0 min | memory cap 6.9 min / time cap 10.0 min | time cap 18.4 min / time cap 10.0 min | time cap – / time cap 10.0 min |
| sine_1k | time cap 11.2 min / optimal 2.9 s | memory cap 8.2 min / time cap 10.0 min | time cap 19.1 min / time cap 10.0 min | time cap 14.4 min / time cap 10.0 min | time cap 12.5 min / time cap 10.0 min |
| car_evaluation | optimal <1 ms / optimal 1 ms | optimal 9 ms / optimal 6 ms | optimal 564 ms / optimal 61 ms | optimal 2.6 s / optimal 196 ms | optimal 6.3 s / optimal 577 ms |
| coupon_bar | optimal <1 ms / optimal <1 ms | optimal <1 ms / optimal 2 ms | optimal 17 ms / optimal 7 ms | optimal 375 ms / optimal 28 ms | optimal 1.9 s / optimal 184 ms |
| compas_binned | optimal <1 ms / optimal <1 ms | optimal <1 ms / optimal <1 ms | optimal <1 ms / optimal 1 ms | optimal 8 ms / optimal 4 ms | optimal 64 ms / optimal 7 ms |
| sine_10k | memory cap 41.7 s / time cap 10.0 min | memory cap 54.6 s / time cap 10.0 min | memory cap 59.7 s / time cap 10.0 min | memory cap 1.0 min / time cap 10.0 min | memory cap 1.0 min / time cap 10.0 min |
| fico_binary | optimal <1 ms / optimal <1 ms | optimal 2 ms / optimal <1 ms | optimal 84 ms / optimal 14 ms | optimal 3.3 s / optimal 251 ms | optimal 31.7 s / optimal 2.9 s |
| compas_processed | optimal 4.9 min / optimal 14 ms | memory cap 4.1 min / memory cap 6.6 min | memory cap 2.2 min / memory cap 7.6 min | memory cap 3.2 min / memory cap 8.2 min | memory cap 3.0 min / memory cap 7.8 min |

Reference: 51 certified optimal, 12 time cap, 12 memory cap. pygosdt: 58 certified optimal, 13 time cap, 4 memory cap (incumbent plus certified lower bound returned).

## Full table

| dataset | λ | C++ objective | py objective | diff | C++ leaves | py leaves | C++ time | py time | C++ nodes | py nodes | C++ stop | py stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chudi | 0.1 | 0.4649 | 0.4649 | 0 | 4 | 4 | 2 ms | 2 ms | 123 | 105 | optimal | optimal |
| chudi | 0.05 | 0.2500 | 0.2500 | 0 | 5 | 5 | 43 ms | 6 ms | 1,455 | 242 | optimal | optimal |
| chudi | 0.02 | 0.1000 | 0.1000 | 0 | 5 | 5 | 567 ms | 10 ms | 3,703 | 405 | optimal | optimal |
| chudi | 0.01 | 0.0500 | 0.0500 | 0 | 5 | 5 | 1.3 s | 11 ms | 4,132 | 423 | optimal | optimal |
| chudi | 0.005 | 0.0250 | 0.0250 | 0 | 5 | 5 | 1.8 s | 11 ms | 4,183 | 423 | optimal | optimal |
| monk_3 | 0.1 | 0.5213 | 0.5213 | 0 | 3 | 3 | <1 ms | <1 ms | 24 | 24 | optimal | optimal |
| monk_3 | 0.05 | 0.3713 | 0.3713 | 0 | 3 | 3 | 5 ms | 3 ms | 656 | 151 | optimal | optimal |
| monk_3 | 0.02 | 0.2430 | 0.2430 | 0 | 6 | 6 | 53 ms | 22 ms | 3,711 | 1,001 | optimal | optimal |
| monk_3 | 0.01 | 0.1556 | 0.1556 | 0 | 9 | 9 | 264 ms | 54 ms | 6,948 | 2,253 | optimal | optimal |
| monk_3 | 0.005 | 0.0946 | 0.0946 | 0 | 14 | 14 | 1.0 s | 94 ms | 9,220 | 3,494 | optimal | optimal |
| monk_1 | 0.1 | 0.4661 | 0.4661 | 0 | 2 | 2 | <1 ms | <1 ms | 29 | 34 | optimal | optimal |
| monk_1 | 0.05 | 0.3387 | 0.3387 | 0 | 5 | 5 | 2 ms | 3 ms | 332 | 146 | optimal | optimal |
| monk_1 | 0.02 | 0.1600 | 0.1600 | 0 | 8 | 8 | 53 ms | 8 ms | 3,822 | 427 | optimal | optimal |
| monk_1 | 0.01 | 0.0800 | 0.0800 | 0 | 8 | 8 | 246 ms | 9 ms | 7,072 | 545 | optimal | optimal |
| monk_1 | 0.005 | 0.0400 | 0.0400 | 0 | 8 | 8 | 771 ms | 10 ms | 8,781 | 583 | optimal | optimal |
| iris | 0.1 | 0.3400 | 0.3400 | 0 | 3 | 3 | 128 ms | 1 ms | 1,571 | 51 | optimal | optimal |
| iris | 0.05 | 0.1900 | 0.1900 | 0 | 3 | 3 | 2.1 s | 2 ms | 16,850 | 63 | optimal | optimal |
| iris | 0.02 | 0.1000 | 0.1000 | 0 | 4 | 3 | 16.4 s | 26 ms | 76,312 | 861 | optimal | optimal |
| iris | 0.01 | 0.0600 | 0.0600 | 0 | 4 | 4 | 2.5 min | 138 ms | 215,881 | 3,812 | optimal | optimal |
| iris | 0.005 | 0.0450 | 0.0383 | -0.0067 | 5 | 5 | 10.0 min | 585 ms | 342,649 | 14,743 | time cap | optimal |
| monk_2 | 0.1 | 0.4787 | 0.4787 | 0 | 1 | 1 | <1 ms | <1 ms | 23 | 23 | optimal | optimal |
| monk_2 | 0.05 | 0.4287 | 0.4287 | 0 | 1 | 1 | 5 ms | 4 ms | 661 | 188 | optimal | optimal |
| monk_2 | 0.02 | 0.3589 | 0.3589 | 0 | 7 | 7 | 75 ms | 47 ms | 4,537 | 2,022 | optimal | optimal |
| monk_2 | 0.01 | 0.2651 | 0.2651 | 0 | 20 | 20 | 444 ms | 129 ms | 8,893 | 4,963 | optimal | optimal |
| monk_2 | 0.005 | 0.1528 | 0.1528 | 0 | 27 | 27 | 1.9 s | 211 ms | 12,052 | 7,525 | optimal | optimal |
| tic-tac-toe | 0.1 | 0.4466 | 0.4466 | 0 | 1 | 1 | 2 ms | 1 ms | 55 | 55 | optimal | optimal |
| tic-tac-toe | 0.05 | 0.3966 | 0.3966 | 0 | 1 | 1 | 163 ms | 21 ms | 4,527 | 1,051 | optimal | optimal |
| tic-tac-toe | 0.02 | 0.3246 | 0.3183 | -0.0063 | 6 | 6 | 19.6 s | 1.7 s | 370,481 | 69,872 | optimal | optimal |
| tic-tac-toe | 0.01 | 0.2508 | 0.2508 | 0 | 9 | 9 | 4.1 min | 11.5 s | 1,862,024 | 426,309 | optimal | optimal |
| tic-tac-toe | 0.005 | – | 0.1543 |  |  | 20 | – | 25.2 s |  | 957,506 | memory cap | optimal |
| gaussian_1k | 0.1 | 0.3970 | 0.3970 | 0 | 1 | 1 | 152 ms | 10 ms | 393 | 175 | optimal | optimal |
| gaussian_1k | 0.05 | 0.2750 | 0.2750 | 0 | 3 | 3 | 2.0 min | 759 ms | 23,204 | 8,340 | optimal | optimal |
| gaussian_1k | 0.02 | 0.3170 | 0.1850 | -0.1320 | 1 | 3 | 10.0 min | 18.9 s | 58,903 | 48,661 | time cap | optimal |
| gaussian_1k | 0.01 | 0.3070 | 0.1550 | -0.1520 | 1 | 3 | 10.1 min | 1.4 min | 79,715 | 81,817 | time cap | optimal |
| gaussian_1k | 0.005 | 0.3020 | 0.1400 | -0.1620 | 1 | 3 | 10.8 min | 4.5 min | 62,470 | 112,006 | time cap | optimal |
| fico_1k | 0.1 | 0.4910 | 0.4910 | 0 | 2 | 2 | 13.6 min | 9.5 s | 17,937 | 95,356 | time cap | optimal |
| fico_1k | 0.05 | 0.3910 | 0.3910 | 0 | 2 | 2 | 11.2 min | 10.0 min | 10,580 | 4,673,540 | time cap | time cap |
| fico_1k | 0.02 | – | 0.3310 |  |  | 2 | – | 10.0 min |  | 5,047,643 | memory cap | time cap |
| fico_1k | 0.01 | 0.3110 | 0.3110 | 0 | 2 | 2 | 18.4 min | 10.0 min | 10,795 | 5,084,652 | time cap | time cap |
| fico_1k | 0.005 | – | 0.3010 |  |  | 2 | – | 10.0 min |  | 5,199,994 | time cap | time cap |
| sine_1k | 0.1 | 0.5950 | 0.5950 | 0 | 1 | 1 | 11.2 min | 2.9 s | 37,422 | 24,046 | time cap | optimal |
| sine_1k | 0.05 | – | 0.5450 |  |  | 1 | – | 10.0 min |  | 264,420 | memory cap | time cap |
| sine_1k | 0.02 | 0.4940 | 0.4940 | 0 | 2 | 2 | 19.1 min | 10.0 min | 4,692 | 186,570 | time cap | time cap |
| sine_1k | 0.01 | 0.4740 | 0.4740 | 0 | 2 | 2 | 14.4 min | 10.0 min | 4,692 | 176,491 | time cap | time cap |
| sine_1k | 0.005 | 0.4620 | 0.4000 | -0.0620 | 3 | 12 | 12.5 min | 10.0 min | 12,642 | 148,584 | time cap | time cap |
| car_evaluation | 0.1 | 0.3998 | 0.3998 | 0 | 1 | 1 | <1 ms | 1 ms | 21 | 21 | optimal | optimal |
| car_evaluation | 0.05 | 0.2947 | 0.2947 | 0 | 3 | 3 | 9 ms | 6 ms | 441 | 118 | optimal | optimal |
| car_evaluation | 0.02 | 0.2047 | 0.2047 | 0 | 3 | 3 | 564 ms | 61 ms | 18,368 | 1,848 | optimal | optimal |
| car_evaluation | 0.01 | 0.1452 | 0.1452 | 0 | 7 | 7 | 2.6 s | 196 ms | 62,947 | 7,189 | optimal | optimal |
| car_evaluation | 0.005 | 0.1063 | 0.1063 | 0 | 9 | 9 | 6.3 s | 577 ms | 125,422 | 20,615 | optimal | optimal |
| coupon_bar | 0.1 | 0.5119 | 0.5119 | 0 | 1 | 1 | <1 ms | <1 ms | 1 | 1 | optimal | optimal |
| coupon_bar | 0.05 | 0.4539 | 0.4539 | 0 | 2 | 2 | <1 ms | 2 ms | 29 | 29 | optimal | optimal |
| coupon_bar | 0.02 | 0.3716 | 0.3716 | 0 | 3 | 3 | 17 ms | 7 ms | 694 | 179 | optimal | optimal |
| coupon_bar | 0.01 | 0.3411 | 0.3411 | 0 | 4 | 4 | 375 ms | 28 ms | 8,775 | 989 | optimal | optimal |
| coupon_bar | 0.005 | 0.3186 | 0.3186 | 0 | 8 | 8 | 1.9 s | 184 ms | 27,611 | 5,708 | optimal | optimal |
| compas_binned | 0.1 | 0.5611 | 0.5611 | 0 | 2 | 2 | <1 ms | <1 ms | 1 | 3 | optimal | optimal |
| compas_binned | 0.05 | 0.4611 | 0.4611 | 0 | 2 | 2 | <1 ms | <1 ms | 1 | 3 | optimal | optimal |
| compas_binned | 0.02 | 0.4011 | 0.4011 | 0 | 2 | 2 | <1 ms | 1 ms | 21 | 21 | optimal | optimal |
| compas_binned | 0.01 | 0.3749 | 0.3749 | 0 | 3 | 3 | 8 ms | 4 ms | 215 | 105 | optimal | optimal |
| compas_binned | 0.005 | 0.3560 | 0.3560 | 0 | 5 | 5 | 64 ms | 7 ms | 1,505 | 354 | optimal | optimal |
| sine_10k | 0.1 | – | 0.5996 |  |  | 1 | – | 10.0 min |  | 305,692 | memory cap | time cap |
| sine_10k | 0.05 | – | 0.5496 |  |  | 1 | – | 10.0 min |  | 280,947 | memory cap | time cap |
| sine_10k | 0.02 | – | 0.5181 |  |  | 2 | – | 10.0 min |  | 499,125 | memory cap | time cap |
| sine_10k | 0.01 | – | 0.4981 |  |  | 2 | – | 10.0 min |  | 501,198 | memory cap | time cap |
| sine_10k | 0.005 | – | 0.4881 |  |  | 2 | – | 10.0 min |  | 377,042 | memory cap | time cap |
| fico_binary | 0.1 | 0.5040 | 0.5040 | 0 | 2 | 2 | <1 ms | <1 ms | 1 | 3 | optimal | optimal |
| fico_binary | 0.05 | 0.4040 | 0.4040 | 0 | 2 | 2 | 2 ms | <1 ms | 25 | 25 | optimal | optimal |
| fico_binary | 0.02 | 0.3440 | 0.3440 | 0 | 2 | 2 | 84 ms | 14 ms | 683 | 428 | optimal | optimal |
| fico_binary | 0.01 | 0.3240 | 0.3240 | 0 | 2 | 2 | 3.3 s | 251 ms | 23,062 | 9,379 | optimal | optimal |
| fico_binary | 0.005 | 0.3087 | 0.3087 | 0 | 4 | 4 | 31.7 s | 2.9 s | 160,078 | 65,876 | optimal | optimal |
| compas_processed | 0.1 | 0.4114 | 0.4114 | 0 | 1 | 1 | 4.9 min | 14 ms | 759 | 175 | optimal | optimal |
| compas_processed | 0.05 | 0.3614 | 0.3614 | 0 | 1 | 1 | – | 6.6 min |  | 2,781,174 | memory cap | memory cap |
| compas_processed | 0.02 | – | 0.3279 |  |  | 2 | – | 7.6 min |  | 2,856,482 | memory cap | memory cap |
| compas_processed | 0.01 | – | 0.3079 |  |  | 2 | – | 8.2 min |  | 2,796,407 | memory cap | memory cap |
| compas_processed | 0.005 | – | 0.2979 |  |  | 2 | – | 7.8 min |  | 2,820,655 | memory cap | memory cap |

## Method notes

- pygosdt: memoised depth-first branch-and-bound with the reference's bounds, Python big-int bitsets, optional numba kernel (identical trees, 1.1–2× faster on wide numeric data).
- Exactness: exhaustive-DP tests on 50 random problems plus 14 pinned real pairs (80 tests).
- The reference's non-exact pairwise feature-exchange bound is not replicated; missing values filled with 0 for both.
- Reproduce: `uv run python benchmarks/run_benchmark.py`, `benchmarks/summarize.py`, `benchmarks/build_report.py`.
