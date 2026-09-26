## Engine fit and lmk's token cap (the temporary server's own log / status)

| condition | KV B/token (engine) | context window | token_budget ("tokens in memory" cap) | working set | baseline |
|---|---|---|---|---|---|
| unset1 | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |
| limit0 | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |
| limit1g | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |
| limit4g | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |
| clear | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |
| unset2 | 65,536 | 262,144 | 979,877 | 77.76 GiB | 14.95 GiB |

## Decode requests (greedy, 256 tokens, prefix cached): medians over kinds x reps, GB (10^9)

During = max over the request (0.25–0.5 s polls); after = last poll 5 s after the response. Peak is per request (reset at its start).

| condition | context | KV bytes | footprint during | MLX active | MLX cache | active+cache | MLX peak | footprint after | active / cache after | ps RSS during / after | decode tok/s, each request | ttft ms | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unset1 | 32k | 2.2 | 28.6 | 22.5 | 7.6 | 27.5 | 23.9 | 23.7 | 19.7 / 3.2 | 20.5 / 18.2 | c 51.2 / c 51.3 / c 51.0 | 1,190 | 3 |
| unset1 | 128k | 8.6 | 55.0 | 37.6 | 26.8 | 53.9 | 43.3 | 36.5 | 25.9 / 9.7 | 26.9 / 18.2 | c 32.8 / c 32.8 / c 32.8 | 3,207 | 3 |
| limit0 | 32k | 2.2 | 22.5 | 22.5 | – | 22.5 | 23.9 | 20.4 | 19.7 / – | 20.3 / 18.0 | c 47.8 / c 47.9 / c 47.7 | 1,167 | 3 |
| limit0 | 128k | 8.6 | 39.7 | 37.6 | – | 37.6 | 43.3 | 26.8 | 25.9 / – | 26.9 / 18.2 | c 28.2 / c 28.2 / c 28.1 | 3,014 | 3 |
| limit1g | 32k | 2.2 | 23.4 | 22.5 | 1.1 | 23.4 | 23.9 | 21.5 | 19.7 / 1.0 | 20.3 / 18.2 | c 47.2 / c 50.6 / c 51.9 | 1,194 | 3 |
| limit1g | 128k | 8.6 | 39.7 | 37.6 | 1.1 | 38.4 | 43.3 | 27.8 | 25.9 / 1.0 | 26.9 / 18.2 | c 32.6 / c 32.5 / c 32.5 | 3,044 | 3 |
| limit4g | 32k | 2.2 | 25.7 | 22.5 | 4.2 | 25.7 | 23.9 | 23.6 | 19.7 / 3.2 | 20.4 / 18.2 | c 51.1 / c 51.2 / c 51.1 | 1,187 | 3 |
| limit4g | 128k | 8.6 | 42.3 | 38.2 | 4.4 | 42.2 | 43.3 | 31.0 | 25.9 / 4.2 | 26.9 / 18.1 | c 32.7 / c 32.6 / c 32.6 | 3,121 | 3 |
| clear | 32k | 2.2 | 26.9 | 22.5 | 5.8 | 25.7 | 23.9 | 20.4 | 19.7 / – | 20.3 / 18.0 | c 51.0 / c 51.0 / c 51.9 | 1,198 | 3 |
| clear | 128k | 8.6 | 53.6 | 37.6 | 25.1 | 51.9 | 43.3 | 26.8 | 25.9 / – | 26.9 / 18.2 | c 32.7 / c 32.7 / c 32.7 | 3,048 | 3 |
| unset2 | 32k | 2.2 | 28.6 | 22.5 | 7.6 | 27.5 | 23.9 | 23.7 | 19.7 / 3.2 | 20.5 / 18.2 | c 51.3 / c 51.3 / c 51.1 | 1,187 | 3 |
| unset2 | 128k | 8.6 | 55.1 | 37.6 | 26.8 | 53.9 | 43.3 | 36.6 | 25.9 / 9.7 | 26.9 / 18.2 | c 32.7 / c 32.7 / c 32.7 | 3,113 | 3 |

## Warm requests (max_tokens 1): cached / prompt tokens shows cold prefill vs restore from the cache; GB

| condition | context | kind | cached / prompt tokens | ttft ms | footprint during | active+cache | MLX peak | footprint after | active / cache after |
|---|---|---|---|---|---|---|---|---|---|
| unset1 | 32k | code | 0 / 32,840 | 111,707 | 29.8 | 28.7 | 26.0 | 20.5 | 19.5 / 0.3 |
| unset1 | 128k | code | 32,768 / 131,144 | 532,017 | 53.0 | 51.7 | 43.3 | 27.1 | 26.1 / 0.1 |
| limit0 | 32k | code | 32,768 / 32,840 | 947 | 21.0 | 19.8 | 21.6 | 20.2 | 19.5 / – |
| limit0 | 128k | code | 131,072 / 131,144 | 2,830 | 37.3 | 31.2 | 36.8 | 26.7 | 25.9 / – |
| limit1g | 32k | code | 32,768 / 32,840 | 1,062 | 22.0 | 20.9 | 21.6 | 20.3 | 19.5 / 0.1 |
| limit1g | 128k | code | 131,072 / 131,144 | 2,893 | 36.0 | 32.0 | 36.8 | 26.7 | 25.9 / 0.1 |
| limit4g | 32k | code | 32,768 / 32,840 | 1,075 | 24.4 | 23.5 | 21.6 | 20.3 | 19.5 / 0.1 |
| limit4g | 128k | code | 131,072 / 131,144 | 2,873 | 37.7 | 34.4 | 36.8 | 26.7 | 25.9 / 0.1 |
| clear | 32k | code | 32,768 / 32,840 | 1,077 | 24.4 | 23.5 | 21.6 | 20.2 | 19.5 / – |
| clear | 128k | code | 131,072 / 131,144 | 2,901 | 47.7 | 44.9 | 36.8 | 26.7 | 25.9 / – |
| unset2 | 32k | code | 32,768 / 32,840 | 1,122 | 24.5 | 24.2 | 21.6 | 20.2 | 19.5 / 0.0 |
| unset2 | 128k | code | 131,072 / 131,144 | 2,936 | 50.9 | 48.6 | 36.8 | 26.7 | 25.9 / 0.0 |
