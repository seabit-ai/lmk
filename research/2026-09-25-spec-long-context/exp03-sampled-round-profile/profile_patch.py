"""Timers around one DFlash speculative round, installed by monkeypatching the engine in this process.

Nothing in lmk/ or the engine is edited: install() swaps the module-level `dflash_round` of
mlx_engine.model_kit.batched_vision.speculative (what SpeculativeGenerationBatch._round calls) for
`profiled_round` below, a copy of the engine's function at ENGINE_COMMIT 42a248c with the mlx_vlm
sampling walk it uses (mlx_vlm 0.6.16 speculative/dflash.py `_sample_dflash_target_walk`, the
non-positioned branch) inlined, plus timers. It also wraps SpeculativeGenerationBatch.next to time
the whole step and the gap between steps.

Two modes, chosen per request through a control file (env LMK_PROF_CONTROL: "<mode> <tag>"):
  wall    the copy does exactly the engine's operations in the engine's order; the timers only
          read the clock. Syncs are where the engine has them: greedy one (`.tolist()` in the
          walk), sampled 1 + one per walked position.
  phases  a sync (mx.eval) after every phase, so each phase's GPU time is its own; the extra syncs
          change the schedule, so phase sums are not the wall round.
          In this mode, the first MICRO_ROUNDS rounds of a request with >= 3 verify positions also
          run a micro-benchmark on that round's real verify logits (after the round's timers stop):
          the engine's per-position walk over every position vs one vectorized draw of all
          positions with one sync, and the pieces of one draw.
Records: one JSON line per round to env LMK_PROF_OUT.
"""
import json
import os
import statistics
import time

import mlx.core as mx

MICRO_ROUNDS = 6
MICRO_REPS = 5

_out = None
_state = {"last_next_end": None, "tag": None, "micro_done": 0}


def _control():
    try:
        with open(os.environ["LMK_PROF_CONTROL"]) as f:
            mode, tag = f.read().split()
    except (OSError, ValueError, KeyError):
        return "wall", "none"
    return mode, tag


def _emit(rec):
    global _out
    if _out is None:
        _out = open(os.environ["LMK_PROF_OUT"], "a", buffering=1)
    _out.write(json.dumps(rec) + "\n")


def _arrays(tree):
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, (list, tuple)):
        return [a for t in tree for a in _arrays(t)]
    return []


def _micro(logits, sampler, draft_count):
    """Per-position engine walk (every position, no early stop) vs one vectorized draw; ms, median of reps."""
    from mlx_vlm.speculative.dflash import _dflash_target_logprobs

    def per_position():
        lp = _dflash_target_logprobs(logits)
        for pos in range(logits.shape[1]):
            t = sampler(lp[:, pos, :])
            mx.eval(t)
            t.reshape(-1).tolist()

    def vectorized():
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = sampler(lp.reshape(-1, lp.shape[-1]))
        mx.eval(t)
        t.tolist()

    def argmax_block():
        t = mx.argmax(logits, axis=-1)
        mx.eval(t)
        t.tolist()

    one = logits[:, 0, :]
    mx.eval(one)
    lp_one = one - mx.logsumexp(one, axis=-1, keepdims=True)
    mx.eval(lp_one)

    def lse_one():
        mx.eval(one - mx.logsumexp(one, axis=-1, keepdims=True))

    from mlx_engine.utils import sampling as es

    def top_p_one():
        mx.eval(es._apply_top_p(lp_one, 0.95))

    def top_k_one():
        mx.eval(es._apply_top_k(lp_one, 20))

    def categorical_one():
        mx.eval(mx.random.categorical(lp_one))

    def sync_only():
        a = mx.array([1])
        mx.eval(a)
        a.tolist()

    out = {}
    for name, fn in [("per_position_all", per_position), ("vectorized_all", vectorized), ("argmax_block", argmax_block),
                     ("lse_one", lse_one), ("top_p_one", top_p_one), ("top_k_one", top_k_one),
                     ("categorical_one", categorical_one), ("sync_only", sync_only)]:
        fn()  # warm (compile for this shape)
        ts = []
        for _ in range(MICRO_REPS):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        out[name] = round(statistics.median(ts), 3)
    out["positions"] = int(logits.shape[1])
    out["draft_count"] = int(draft_count)
    return out


def install():
    from mlx_engine.model_kit.batched_vision import speculative as S
    from mlx_vlm.speculative.common import generation_stream
    from mlx_vlm.speculative.dflash import _dflash_target_logprobs, _supports_positioned_target_sampling

    orig_next = S.SpeculativeGenerationBatch.next
    orig_round = S.dflash_round

    def profiled_round(model, drafter, prompt_cache, bonus, context, draft_cache, sampler, budget, block_size,
                       truncate, emitted, rope_deltas=None, processors=None, history=None,
                       stop=lambda token: False):
        if processors or not (getattr(sampler, "greedy", False) or not _supports_positioned_target_sampling(sampler)):
            _state["pending"] = {"tag": _control()[1], "mode": "unprofiled", "why": "processors or positioned sampler"}
            return orig_round(model, drafter, prompt_cache, bonus, context, draft_cache, sampler, budget, block_size,
                              truncate, emitted, rope_deltas=rope_deltas, processors=processors, history=history, stop=stop)
        mode, tag = _control()
        if tag != _state["tag"]:
            _state["tag"], _state["micro_done"] = tag, 0
        phases = mode == "phases"
        pc = time.perf_counter
        ms = lambda a, b: round((b - a) * 1000, 3)
        rec = {"tag": tag, "mode": mode, "block": int(block_size)}

        def barrier(*trees):
            if phases:
                mx.eval(*[a for t in trees for a in _arrays(t)])

        t_start = pc()
        lm = S._language_model(model)
        greedy = bool(getattr(sampler, "greedy", False))
        rec["greedy"] = greedy
        rec["positioned_sampler"] = _supports_positioned_target_sampling(sampler)
        rec["processors"] = bool(processors)
        draft_sampler = (lambda logits: mx.argmax(logits, axis=-1).astype(S.TOKEN_DTYPE)) if greedy else sampler
        draft_tokens = drafter.model.draft_block(bonus, context, draft_cache, block_size, draft_sampler, S.TOKEN_DTYPE)
        t_draft_built = pc()
        barrier(draft_tokens)
        t_draft = pc()
        verify_input = mx.concatenate([mx.array([[bonus]], dtype=S.TOKEN_DTYPE), draft_tokens.astype(S.TOKEN_DTYPE)], axis=1)
        with mx.stream(generation_stream):
            verify = S._verify_block(lm, verify_input, prompt_cache, rope_deltas, greedy, drafter=drafter)
        t_verify_built = pc()
        barrier(verify.logits, verify.hidden, verify.gdn_states)
        t_verify = pc()
        rec.update(draft_build_ms=ms(t_start, t_draft_built), draft_ms=ms(t_draft_built, t_draft),
                   verify_build_ms=ms(t_draft, t_verify_built), verify_ms=ms(t_verify_built, t_verify))

        if greedy:
            barrier(verify.target_tokens)
            t_argmax = pc()
            # mlx_vlm common._speculative_walk, inlined to time its two host syncs
            n_draft = int(draft_tokens.shape[1])
            draft_row = draft_tokens.reshape(-1).tolist()[:n_draft]
            t_sync1 = pc()
            target_row = verify.target_tokens.reshape(-1).tolist()
            t_sync2 = pc()
            accepted = n_draft
            for i, (d, t) in enumerate(zip(draft_row, target_row)):
                if d != t:
                    accepted = i
                    break
            new_tokens = (draft_row[:accepted] + target_row[accepted:accepted + 1])[:budget]
            t_walk = pc()
            rec.update(argmax_ms=ms(t_verify, t_argmax), sync_draft_tolist_ms=ms(t_argmax, t_sync1),
                       sync_target_tolist_ms=ms(t_sync1, t_sync2), walk_host_ms=ms(t_sync2, t_walk),
                       walk_ms=ms(t_verify, t_walk), walk_positions=1)
        else:
            # mlx_vlm dflash._sample_dflash_target_walk, non-positioned branch, one row, inlined with timers
            batch, length, _ = verify.logits.shape
            draft_count = int(draft_tokens.shape[1])
            draft_rows = draft_tokens.tolist()
            t_sync1 = pc()
            logprobs = _dflash_target_logprobs(verify.logits)
            t_lse_built = pc()
            barrier(logprobs)
            t_lse = pc()
            per_pos = []
            result = None
            for position in range(length):
                a = pc()
                target_tokens = sampler(logprobs[:, position, :])
                b = pc()
                mx.eval(target_tokens)
                c = pc()
                target_rows = [int(token) for token in target_tokens.reshape(-1).tolist()]
                d = pc()
                per_pos.append({"build": ms(a, b), "eval": ms(b, c), "tolist": ms(c, d)})
                if position < draft_count and target_rows[0] == draft_rows[0][position]:
                    continue
                tokens = draft_rows[0][:position]
                if len(tokens) < budget:
                    tokens.append(target_rows[0])
                result = (position, tokens[:budget])
                break
            if result is None:
                result = (draft_count, draft_rows[0][:budget])
            accepted, new_tokens = result
            t_walk = pc()
            rec.update(sync_draft_tolist_ms=ms(t_verify, t_sync1), lse_build_ms=ms(t_sync1, t_lse_built),
                       lse_ms=ms(t_lse_built, t_lse), positions=per_pos, walk_positions=len(per_pos),
                       walk_ms=ms(t_verify, t_walk))

        S._record_speculative_round(drafter.model, accepted, block_size - 1)
        cut, reason = truncate(0, [int(t) for t in new_tokens])
        kept = len(cut) - 1
        t_book = pc()
        if kept < block_size - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, verify.gdn_states, kept, block_size)
        t_rb_built = pc()
        if phases:
            barrier([c.state for c in prompt_cache if c is not None and not c.is_trimmable()])
        t_rb = pc()
        result = S.RoundResult(new_tokens=[cut], finish=[reason], hidden=verify.hidden[:, : kept + 1, :], accepted=[kept])
        t_end = pc()
        rec.update(accepted=int(accepted), kept=int(kept), bookkeeping_ms=ms(t_walk, t_book),
                   rollback_build_ms=ms(t_book, t_rb_built), rollback_ms=ms(t_rb_built, t_rb),
                   rolled_back=kept < block_size - 1, round_ms=ms(t_start, t_end))
        if phases and not greedy and length >= 3 and _state["micro_done"] < MICRO_ROUNDS:
            _state["micro_done"] += 1
            rec["micro"] = _micro(verify.logits, sampler, draft_count)
        elif phases and greedy and verify.logits.shape[1] >= 3 and _state["micro_done"] < MICRO_ROUNDS:
            _state["micro_done"] += 1
            rec["micro"] = _micro(verify.logits, _greedy_free_sampler(), int(draft_tokens.shape[1]))
        _state["pending"] = rec
        return result

    def profiled_next(self):
        t0 = time.perf_counter()
        gap = None if _state["last_next_end"] is None else round((t0 - _state["last_next_end"]) * 1000, 3)
        _state["pending"] = None
        out = orig_next(self)
        t1 = time.perf_counter()
        _state["last_next_end"] = t1
        rec = _state.get("pending")
        if rec is not None:
            rec.update(step_ms=round((t1 - t0) * 1000, 3), gap_before_ms=gap, t=time.time())
            _emit(rec)
        return out

    S.dflash_round = profiled_round
    S.SpeculativeGenerationBatch.next = profiled_next


def _greedy_free_sampler():
    """For micro-benchmarking a greedy round's logits with the model's default sampler (temp 1.0 /
    top_p 0.95 / top_k 20, lmk's defaults for this model): the cost of a draw does not depend on
    which request produced the logits."""
    from mlx_engine.utils.sampling import create_sampler
    return create_sampler(1.0, 0.95, None, None, 20)
