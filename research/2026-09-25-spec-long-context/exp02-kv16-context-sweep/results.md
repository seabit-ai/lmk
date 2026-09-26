## Decode: kv8 (exp01) vs kv16 (exp02)

| context | kind | mode | off ms/step kv8 → kv16 | off tok/s kv8 → kv16 (kv16 min–max) | on tok/s kv8 → kv16 (kv16 min–max) | speedup kv8 → kv16 | accepted kv8 → kv16 | round ms kv8 (est.) → kv16 (exact) | greedy text off = on, kv16 |
|---|---|---|---|---|---|---|---|---|---|
| 8k | code | greedy | 28.3 → 27.0 | 35.3 → 37.0 (37.0–37.0) | 53.0 → 53.7 (50.1–56.4) | 1.50× → 1.45× | 73% → 74% | 71 → 70 | no (1 off / 1 on distinct) |
| 8k | prose | greedy | 28.3 → 27.0 | 35.4 → 37.0 (37.0–37.0) | 39.4 → 43.9 (43.6–43.9) | 1.12× → 1.19× | 56% → 59% | 54 → 49 | no (1 off / 1 on distinct) |
| 32k | code | greedy | 36.2 → 31.0 | 27.6 → 32.2 (32.2–32.3) | 41.6 → 49.6 (49.3–49.7) | 1.51× → 1.54× | 83% → 81% | 94 → 77 | no (1 off / 1 on distinct) |
| 32k | prose | greedy | 36.1 → 31.1 | 27.7 → 32.2 (32.0–32.3) | 35.3 → 37.7 (37.6–38.0) | 1.27× → 1.17× | 63% → 63% | 64 → 60 | no (1 off / 1 on distinct) |
| 64k | code | greedy | 46.3 → 35.9 | 21.6 → 27.8 (27.8–27.9) | 30.9 → 41.6 (41.5–41.6) | 1.43× → 1.49× | 81% → 79% | 123 → 90 | no (1 off / 1 on distinct) |
| 64k | prose | greedy | 46.6 → 36.0 | 21.5 → 27.8 (27.8–27.8) | 26.6 → 32.3 (32.3–32.3) | 1.24× → 1.16× | 63% → 62% | 86 → 69 | no (1 off / 1 on distinct) |
| 128k | code | greedy | 65.1 → 45.2 | 15.4 → 22.1 (22.1–22.1) | 19.7 → 31.2 (31.1–31.2) | 1.28× → 1.41× | 71% → 76% | 179 → 128 | no (1 off / 1 on distinct) |
| 128k | prose | greedy | 65.6 → 45.2 | 15.2 → 22.1 (22.1–22.1) | 18.8 → 24.0 (23.9–24.0) | 1.23× → 1.08× | 63% → 59% | 120 → 94 | no (1 off / 1 on distinct) |
| 32k | code | sampled | 36.5 → 31.5 | 27.4 → 31.8 (31.7–31.8) | 38.3 → 43.7 (40.6–47.6) | 1.40× → 1.37× | 74% → 73% | 70 → 75 | — |
| 32k | prose | sampled | 36.4 → 31.5 | 27.4 → 31.8 (31.7–31.8) | 28.1 → 33.7 (32.3–36.5) | 1.02× → 1.06× | 54% → 54% | 73 → 60 | — |

## Memory (kv16 measured; kv8's process bytes were not measured in exp01)

| context (prompt tokens) | KV bytes kv8 → kv16 (engine bytes/token × tokens) | kv16 lmk_gpu_bytes max during decode, off / on | kv16 MLX peak in use (process peak so far), off / on |
|---|---|---|---|
| 8k (8522) | 0.30 → 0.56 GB | 18.87 / 20.87 GB | 20.39 / 19.11 GB |
| 32k (33098) | 1.15 → 2.17 GB | 25.48 / 27.67 GB | 24.58 / 23.94 GB |
| 64k (65866) | 2.29 → 4.32 GB | 34.24 / 36.65 GB | 30.07 / 30.38 GB |
| 128k (131402) | 4.57 → 8.61 GB | 51.95 / 54.19 GB | 42.17 / 43.27 GB |

lmk_gpu_bytes = MLX active + MLX buffer cache (lmk/engine.py gpu_memory_bytes): what the process holds, including freed buffers MLX keeps for reuse — not the KV alone.

runs discarded because the resident lmk was busy: exp01 3, exp02 0
