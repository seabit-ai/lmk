"""Experiment-side probe loaded into a temporary `lmk serve` through PYTHONPATH (Python imports
`sitecustomize` at startup). Touches no file under lmk/ or the engine; everything is env-driven and
does nothing when LMK_EXP_PROBE is unset.

  LMK_EXP_PROBE=<jsonl>        append MLX memory readings: a sample every 0.25 s, plus one line at the
                               start and at the end of every generation (lmk.engine.MlxEngine.generate)
  LMK_EXP_CACHE_LIMIT=<bytes>  mx.set_cache_limit(<bytes>) before the engine loads (exp02)
  LMK_EXP_CLEAR_AT_END=1       mx.clear_cache() when a generation's stream ends (exp02)

Each generation resets MLX's peak counter at its start, so the end line's `peak` is that request's peak
(cache restore + suffix prefill + decode)."""
import json
import os
import threading
import time

_PATH = os.environ.get("LMK_EXP_PROBE")

if _PATH:
    import mlx.core as mx

    _lock = threading.Lock()

    def _write(rec):
        rec["t_ms"] = int(time.time() * 1000)
        with _lock, open(_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _reading():
        return {"active": mx.get_active_memory(), "cache": mx.get_cache_memory(), "peak": mx.get_peak_memory()}

    _limit = os.environ.get("LMK_EXP_CACHE_LIMIT")
    _default = mx.set_cache_limit(0)          # returns the default; put it straight back
    mx.set_cache_limit(_default)
    if _limit:
        mx.set_cache_limit(int(_limit))
    _clear = os.environ.get("LMK_EXP_CLEAR_AT_END") == "1"
    _write({"ev": "ProbeStart", "pid": os.getpid(), "default_cache_limit": _default,
            "cache_limit": int(_limit) if _limit else None, "clear_at_end": _clear,
            "device": {k: v for k, v in mx.device_info().items() if isinstance(v, (int, float, str))}})

    def _sampler():
        while True:
            try:
                _write({"ev": "S", **_reading()})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)

    threading.Thread(target=_sampler, daemon=True, name="exp-probe").start()

    import lmk.engine as _engine

    _orig_generate = _engine.MlxEngine.generate

    def _generate(self, prompt_text, **kw):
        ref = kw.get("request_id")
        mx.reset_peak_memory()
        _write({"ev": "GenStart", "req": ref, **_reading()})
        gen = _orig_generate(self, prompt_text, **kw)

        def pieces():
            try:
                yield from gen.pieces
            finally:
                _write({"ev": "GenEnd", "req": ref, **_reading()})
                if _clear:
                    mx.clear_cache()
                    _write({"ev": "Cleared", "req": ref, **_reading()})

        return _engine.Generation(pieces=pieces(), stats=gen.stats)

    _engine.MlxEngine.generate = _generate
