"""Re-lay the 122B MTP drafter's per-expert weights into mlx's stacked switch_mlp layout.

The split tool writes the original per-expert tensors (layers.0.mlp.experts.<i>.{gate,up,down}_proj.weight,
256 x 3); mlx-vlm's qwen3_5_mtp drafter only converts the fused HF layout (experts.gate_up_proj) and
otherwise expects layers.0.mlp.switch_mlp.{gate,up,down}_proj.weight stacked along a new axis 0 —
exactly what the MLX conversions of the main model carry. Values are unchanged, only stacked.
usage: restack_experts.py <drafter dir>   (rewrites model.safetensors in place, keeps the metadata)
"""
import re, sys
from pathlib import Path
import mlx.core as mx

d = Path(sys.argv[1])
weights, meta = mx.load(str(d / "model.safetensors"), return_metadata=True)
pat = re.compile(r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
groups: dict[tuple[str, str], dict[int, mx.array]] = {}
for k in list(weights):
    m = pat.match(k)
    if m:
        groups.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = weights.pop(k)
for (prefix, proj), by_idx in groups.items():
    n = len(by_idx)
    assert sorted(by_idx) == list(range(n)), (prefix, proj, n)
    weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack([by_idx[i] for i in range(n)], axis=0)
    print(f"{prefix}.switch_mlp.{proj}.weight  <- {n} experts  {weights[f'{prefix}.switch_mlp.{proj}.weight'].shape}", flush=True)
mx.save_safetensors(str(d / "model.safetensors"), weights, metadata=meta)
print("wrote", d / "model.safetensors", "keys", len(weights))
