## Engine fit and lmk's token cap (the temporary server's own log / status)

| condition | KV B/token (engine) | context window | token_budget ("tokens in memory" cap) | working set | baseline |
|---|---|---|---|---|---|
| unset1 | 34,816 | 262,144 | 1,844,474 | 77.76 GiB | 14.95 GiB |
| limit1g | 34,816 | 262,144 | 1,844,474 | 77.76 GiB | 14.95 GiB |
| unset2 | 34,816 | 262,144 | 1,844,474 | 77.76 GiB | 14.95 GiB |

## Decode requests (greedy, 256 tokens, prefix cached): medians over kinds x reps, GB (10^9)

During = max over the request (0.25–0.5 s polls); after = last poll 5 s after the response. Peak is per request (reset at its start).

| condition | context | KV bytes | footprint during | MLX active | MLX cache | active+cache | MLX peak | footprint after | active / cache after | ps RSS during / after | decode tok/s, each request | ttft ms | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unset1 | 32k | 1.2 | 24.5 | 19.8 | 4.6 | 23.4 | 20.9 | 21.6 | 18.6 / 2.1 | 19.3 / 18.2 | c 38.7 / c 41.3 / c 43.5 | 1,265 | 3 |
| unset1 | 128k | 4.6 | 39.0 | 27.0 | 15.0 | 37.7 | 31.2 | 28.5 | 21.9 / 5.7 | 23.1 / 18.2 | c 20.2 / c 20.2 / c 20.2 | 3,726 | 3 |
| limit1g | 32k | 1.2 | 21.2 | 19.8 | 1.1 | 20.5 | 20.9 | 20.5 | 18.6 / 1.0 | 18.9 / 18.0 | c 37.1 / c 36.9 / c 41.6 | 1,235 | 3 |
| limit1g | 128k | 4.6 | 35.1 | 29.6 | 1.3 | 29.6 | 31.2 | 23.8 | 21.9 / 1.0 | 23.1 / 18.2 | c 19.6 / c 20.2 / c 20.0 | 3,449 | 3 |
| unset2 | 32k | 1.2 | 24.5 | 19.9 | 4.6 | 23.4 | 20.9 | 21.6 | 18.6 / 2.1 | 19.3 / 18.0 | c 43.6 / c 43.7 / c 41.2 | 1,253 | 3 |
| unset2 | 128k | 4.6 | 39.0 | 27.0 | 15.0 | 37.7 | 31.2 | 28.4 | 21.9 / 5.7 | 23.0 / 18.2 | c 20.2 / c 20.2 / c 20.2 | 3,679 | 3 |

## Warm requests (max_tokens 1): cached / prompt tokens shows cold prefill vs restore from the cache; GB

| condition | context | kind | cached / prompt tokens | ttft ms | footprint during | active+cache | MLX peak | footprint after | active / cache after |
|---|---|---|---|---|---|---|---|---|---|
| unset1 | 32k | code | 0 / 32,840 | 113,174 | 28.0 | 26.9 | 25.0 | 19.5 | 18.4 / 0.3 |
| unset1 | 128k | code | 32,768 / 131,144 | 633,925 | 44.7 | 43.9 | 39.1 | 22.9 | 22.0 / 0.1 |
| limit1g | 32k | code | 32,768 / 32,840 | 1,103 | 20.9 | 19.9 | 19.6 | 19.3 | 18.4 / 0.1 |
| limit1g | 128k | code | 131,072 / 131,144 | 3,370 | 31.3 | 24.5 | 27.7 | 22.8 | 21.9 / 0.1 |
| unset2 | 32k | code | 32,768 / 32,840 | 1,152 | 22.0 | 20.8 | 19.6 | 19.2 | 18.4 / 0.0 |
| unset2 | 128k | code | 131,072 / 131,144 | 3,441 | 36.7 | 34.6 | 27.7 | 22.7 | 21.9 / 0.0 |
