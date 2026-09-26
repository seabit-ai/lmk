| context (prompt tokens) | kind | mode | off tok/s median (min–max) | on tok/s median (min–max) | speedup | accepted / drafted (rate) | finish:tokens off / on | greedy text off = on |
|---|---|---|---|---|---|---|---|---|
| 8k (8264) | code | greedy | 35.3 (35.3–35.3) | 53.0 (48.5–53.3) | 1.50× | 561 / 771 (73%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 8k (8266) | prose | greedy | 35.4 (35.3–35.4) | 39.4 (38.5–41.2) | 1.12× | 405 / 726 (56%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 32k (32840) | code | greedy | 27.6 (27.5–27.6) | 41.6 (41.3–43.7) | 1.51× | 570 / 687 (83%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 32k (32842) | prose | greedy | 27.7 (27.7–27.9) | 35.3 (35.2–35.3) | 1.27× | 426 / 681 (63%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 64k (65608) | code | greedy | 21.6 (21.3–21.8) | 30.9 (30.9–30.9) | 1.43× | 564 / 699 (81%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 64k (65610) | prose | greedy | 21.5 (21.3–21.8) | 26.6 (26.5–26.6) | 1.24× | 429 / 681 (63%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 128k (131144) | code | greedy | 15.4 (15.0–15.4) | 19.7 (19.7–19.8) | 1.28× | 549 / 768 (71%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 128k (131146) | prose | greedy | 15.2 (14.9–15.3) | 18.8 (18.8–18.8) | 1.23× | 426 / 681 (63%) | length:256 / length:256 | no (1 off / 1 on distinct) |
| 32k (32840) | code | sampled | 27.4 (27.2–27.4) | 38.3 (37.3–39.2) | 1.40× | 493 / 665 (74%) | length:256 / length:256 | — |
| 32k (32842) | prose | sampled | 27.4 (27.4–27.5) | 28.1 (27.8–29.9) | 1.02× | 397 / 739 (54%) | length:256 / length:256 | — |

runs discarded because the resident lmk was busy: 3
