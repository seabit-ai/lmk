## Engine fit and lmk's token cap (the temporary server's own log / status)

| condition | KV B/token (engine) | context window | token_budget ("tokens in memory" cap) | working set | baseline |
|---|---|---|---|---|---|
| unset | 34,816 | 262,144 | 1,844,474 | 77.76 GiB | 14.95 GiB |
| limit4g | 34,816 | 262,144 | 1,844,474 | 77.76 GiB | 14.95 GiB |

## Decode requests (greedy, 256 tokens, prefix cached): medians over kinds x reps, GB (10^9)

During = max over the request (0.25–0.5 s polls); after = last poll 5 s after the response. Peak is per request (reset at its start).

| condition | context | KV bytes | footprint during | MLX active | MLX cache | active+cache | MLX peak | footprint after | active / cache after | ps RSS during / after | decode tok/s, each request | ttft ms | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unset | 32k | 1.2 | 24.5 | 20.2 | 4.6 | 23.4 | 20.9 | 21.6 | 18.6 / 2.1 | 19.3 / 18.2 | c 42.2 / c 43.5 / c 41.3 | 1,266 | 3 |
| unset | 128k | 4.6 | 39.0 | 30.2 | 14.9 | 37.7 | 31.2 | 28.4 | 21.9 / 5.6 | 23.1 / 18.2 | c 20.5 / c 20.5 / c 20.5 | 3,753 | 3 |
| limit4g | 32k | 1.2 | 24.2 | 20.3 | 4.1 | 23.1 | 20.9 | 21.6 | 18.6 / 2.2 | 19.3 / 18.0 | c 41.2 / c 43.6 / c 43.6 | 1,236 | 3 |
| limit4g | 128k | 4.6 | 36.2 | 27.0 | 4.3 | 31.1 | 31.2 | 26.9 | 21.9 / 4.2 | 23.1 / 18.2 | c 20.5 / c 20.3 / c 20.4 | 3,447 | 3 |

## Warm requests (max_tokens 1): cached / prompt tokens shows cold prefill vs restore from the cache; GB

| condition | context | kind | cached / prompt tokens | ttft ms | footprint during | active+cache | MLX peak | footprint after | active / cache after |
|---|---|---|---|---|---|---|---|---|---|
| unset | 32k | code | 0 / 32,840 | 113,183 | 28.0 | 26.8 | 25.0 | 19.5 | 18.4 / 0.3 |
| unset | 128k | code | 32,768 / 131,144 | 634,162 | 45.4 | 44.3 | 39.1 | 22.9 | 22.0 / 0.1 |
| limit4g | 32k | code | 32,768 / 32,840 | 1,135 | 21.6 | 21.1 | 19.6 | 19.2 | 18.4 / 0.0 |
| limit4g | 128k | code | 131,072 / 131,144 | 3,183 | 30.1 | 27.0 | 27.7 | 22.7 | 21.9 / 0.0 |
