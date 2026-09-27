import runpy, sys
import mlx_engine.model_kit.patches.qwen3_5 as P
import mlx_vlm.models.qwen3_5.language as L
orig = L.LanguageModel.__call__
def traced(self, inputs, *a, **kw):
    if inputs is not None and inputs.shape[-1] > 1:
        desc = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in kw.items() if k not in ("cache",)}
        rd = kw.get("rope_deltas"); pid = kw.get("position_ids")
        c = kw.get("cache")
        off = None
        if c is not None:
            c0 = c[self.model.fa_idx]
            off = getattr(c0, "_idx", None) if hasattr(c0, "_idx") else getattr(c0, "offset", None)
        print(f"  LM call n={inputs.shape[-1]} kw={desc} rope_deltas={None if rd is None else tuple(rd.shape)} "
              f"pos_ids={None if pid is None else tuple(pid.shape)} "
              f"fa_offset={off} self._rope_deltas={None if self._rope_deltas is None else tuple(self._rope_deltas.shape)}", flush=True)
    return orig(self, inputs, *a, **kw)
L.LanguageModel.__call__ = traced
sys.argv = [sys.argv[1], "draft"]
runpy.run_path(sys.argv[0], run_name="__main__")
