## Fitted lines (96 GB Mac, one request at a time; t = tokens in the request)

worst held = max(footprint, MLX peak + 1.12 GB non-MLX). Uncapped: exp01 decode requests at 8k/32k/64k/128k. Capped (4 GiB): the kv16 / kv8 capped runs at 32k/128k.

| model | bits | points (tokens: GB) | worst held (GB) = a + b·t | b ÷ KV B/token |
|---|---|---|---|---|
| capped | kv8 | 33k: 24.2, 131k: 36.5 | 20.0 + 126 kB × t | 3.6× |
| capped | kv16 | 33k: 25.8, 131k: 44.4 | 19.5 + 189 kB × t | 2.9× |
| uncapped | kv8 | 8k: 20.7, 8k: 21.0, 33k: 24.4, 33k: 24.6, 65k: 29.4, 65k: 29.4, 131k: 39.2, 131k: 39.2 | 19.6 + 149 kB × t | 4.3× |
| uncapped | kv16 | 8k: 22.1, 8k: 22.2, 33k: 28.4, 33k: 28.7, 65k: 37.6, 65k: 37.5, 131k: 55.1, 131k: 55.4 | 19.8 + 270 kB × t | 4.1× |

## Per Mac size (computed; only 96 GB was measured). Capped: MLX cache limit 4 GiB

working set = GB × 0.81 GiB (lmk's GPU_SHARE_OF_MEMORY). Window and cap: lmk's formulas (`context_on`, `engine.token_budget`). 'At full window': the lines above evaluated at the formula window (beyond 131k extrapolated), minus the working set. 'Largest context': where the line meets the working set — also what one request could really use of the tokens-in-memory cap.

| Mac | bits | window (formula) | tokens-in-memory cap (formula) | at full window vs working set, uncapped (GB) | capped | largest context under the working set, uncapped | capped |
|---|---|---|---|---|---|---|---|
| 16 GB | kv8 | does not load | – | – | – | – | – |
| 16 GB | kv16 | does not load | – | – | – | – | – |
| 24 GB | kv8 | 23k | 46k | +2.1 | +2.0 | 9k | 7k |
| 24 GB | kv16 | 16k | 24k | +3.2 | +1.6 | 4k | 7k |
| 32 GB | kv8 | 123k | 246k | +10.1 | +7.6 | 55k | 62k |
| 32 GB | kv16 | 85k | 131k | +14.9 | +7.8 | 30k | 44k |
| 48 GB | kv8 | 262k | 645k | +16.9 | +11.2 | 149k | 173k |
| 48 GB | kv16 | 224k | 343k | +38.4 | +20.1 | 81k | 117k |
| 64 GB | kv8 | 262k | 1.05M | +3.0 | -2.7 | 242k | 262k |
| 64 GB | kv16 | 262k | 555k | +34.9 | +13.5 | 133k | 191k |
| 96 GB | kv8 | 262k | 1.84M | -24.8 | -30.6 | 262k | 262k |
| 96 GB | kv16 | 262k | 980k | +7.0 | -14.4 | 236k | 262k |
| 128 GB | kv8 | 262k | 2.64M | -52.7 | -58.4 | 262k | 262k |
| 128 GB | kv16 | 262k | 1.40M | -20.8 | -42.2 | 262k | 262k |
| 192 GB | kv8 | 262k | 4.24M | -108.3 | -114.1 | 262k | 262k |
| 192 GB | kv16 | 262k | 2.25M | -76.5 | -97.8 | 262k | 262k |
