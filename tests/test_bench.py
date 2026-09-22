from lmk import bench


def fake_stream(script):
    """A StreamFn that replays (ms, chunk) lists per call, and records the bodies it was given."""
    calls = []

    def run(body):
        calls.append(body)
        return script[len(calls) - 1]
    run.calls = calls
    return run


def chunks(first_ms, total_ms, prompt, cached, completion, restore_ms=None):
    return [(first_ms, {"choices": [{"delta": {"content": "x"}}]}),
            (total_ms, {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                                 "prompt_tokens_details": {"cached_tokens": cached}},
                        "lmk": {"restore_ms": restore_ms}})]


def test_the_three_probes_and_what_is_computed_from_them():
    stream = fake_stream([chunks(300, 500, 20, 0, 8),                # warm-up
                          chunks(12_775, 14_143, 4060, 0, 32),      # cold: 4060 uncached in 12.8 s
                          chunks(450, 1_800, 4060, 4060, 32, restore_ms=72),  # hit: all cached, back from disk in 72 ms
                          chunks(422, 17_869, 66, 0, 400)])         # decode: 400 tokens in 17.4 s after ttft
    r = bench.run_bench(stream, "m", seed=7)
    assert round(r.cold_prefill_tok_s) == 318 and r.hit_first_token_s == 0.45 and round(r.decode_tok_s, 1) == 22.9
    assert round(r.hit_tok_s) == 56389                                           # 4,060 cached tokens in 72 ms
    bodies = stream.calls
    assert all(b["temperature"] == 0 and b["stream"] is True and b["model"] == "m" for b in bodies)
    assert bodies[1]["messages"] == bodies[2]["messages"]                        # the hit is the same request
    assert bodies[1]["messages"][0]["content"].startswith(f"Benchmark run {bench.nonce_for(7)}.\n") and r.seed == 7
    assert bodies[3]["max_tokens"] == bench.DECODE_TOKENS and r.warmup.first_token_ms == 300


def test_the_nonce_is_ten_digits_from_the_seed_so_the_token_count_does_not_move():
    n = bench.nonce_for(1)
    assert len(n) == 10 and n.isdigit()
    assert bench.nonce_for(1) == n and bench.nonce_for(2) != n
    # the first version made every digit from a fresh Random(seed): ten possible prompts in all,
    # and a "cold" run hit the cache after a few tries
    assert len({bench.nonce_for(s) for s in range(200)}) == 200


def test_a_reused_seed_that_hits_the_cache_is_reported_as_such_not_as_a_cold_number():
    stream = fake_stream([chunks(300, 500, 20, 0, 8), chunks(1_000, 1_800, 4060, 3840, 32, 70),
                          chunks(1_000, 1_800, 4060, 3840, 32, 70), chunks(422, 17_869, 66, 0, 400)])
    r = bench.run_bench(stream, "m", seed=7)
    assert not r.cold_was_cold
    assert "seed 7 reused" in bench.human_block(r) and "tokens/s\n" not in bench.human_block(r).split("\n")[0]
    row = bench.markdown_row(r, {"chip": "c", "memory_gb": 1}, {"model": {"id": "m", "context_length": 1}}, "d")
    assert "| — (seed reused: 3,840 cached) |" in row


def test_markdown_row_carries_the_conditions_next_to_the_numbers():
    r = bench.BenchResult(seed=1, warmup=bench.Probe(20, 0, 8, 300, 500), cold=bench.Probe(4060, 0, 32, 12_775, 14_143), hit=bench.Probe(4060, 4060, 32, 450, 1800, restore_ms=72),
                          decode=bench.Probe(66, 0, 400, 422, 17_869))
    row = bench.markdown_row(r, {"chip": "Apple M3 Ultra", "memory_gb": 96},
                             {"model": {"id": "qwen3.8-27b-4bit", "context_length": 262144}, "build": "v0.1.0",
                              "engine": "08f0c07abcdef"}, "2026-09-22")
    assert row == ("| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit | 262,144 | 318 tok/s (4,060 tokens) | "
                   "56k tok/s (4,060 cached; first token 0.45 s) | 22.9 tok/s | v0.1.0 | 08f0c07 | 2026-09-22 |")
    assert row.count("|") == bench.ROW_HEADER.splitlines()[0].count("|")


def test_human_block_is_three_numbers_and_flags_a_slow_warm_up_only_when_it_happened():
    fast = bench.BenchResult(seed=1, warmup=bench.Probe(20, 0, 8, 300, 500), cold=bench.Probe(4060, 0, 32, 12_775, 14_143),
                             hit=bench.Probe(4060, 3840, 32, 1_010, 1_800, restore_ms=72), decode=bench.Probe(66, 0, 400, 422, 17_869))
    assert bench.human_block(fast) == ("  prefill              318 tokens/s\n"
                                       "  cached prefill       53k tokens/s   (3,840 of 4,060 tokens from disk; "
                                       "first token after 1.01 s, the rest is the last partial block computed)\n"
                                       "  decode              22.9 tokens/s")
    slow = bench.BenchResult(seed=1, warmup=bench.Probe(20, 0, 8, 37_000, 37_500), cold=fast.cold, hit=fast.hit, decode=fast.decode)
    assert bench.human_block(slow).endswith("(the warm-up request took 37 s: the weights had to be paged back in; not counted)")
