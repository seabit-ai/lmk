import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache
from mlx_engine.model_kit.patches.rotating_sink_mask import apply_rotating_sink_mask_patch

def kv(start, n):
    k = mx.arange(start, start + n, dtype=mx.float32).reshape(1, 1, n, 1); return k, k
def step_chunk(c, n):
    mask = c.make_mask(n); k, _ = c.update_and_fetch(*kv(c.offset, n)); return mask, k
def visible(mask, k, row=-1):
    cols = [int(x) for x in k[0, 0, :, 0].tolist()]
    if mask is None or isinstance(mask, str): return set(cols)   # None / causal: last row sees all
    r = mask[row]; return set(cols[i] for i in range(len(cols)) if bool(r[i]))
def expected(P, max_size, keep):
    # what decode shows: the keep sinks + the max_size-keep most recent tokens (incl. self)
    return set(range(min(keep, P + 1))) | set(range(max(0, P - (max_size - keep) + 1), P + 1))

def run(patched):
    bad = []; cases = 0; engaged = 0
    for max_size in (8, 16, 32):
      for keep in (0, 1, 4, 7):
        if keep >= max_size: continue
        for step in (3, 9, 16):
          for history in (0, 5, 19, 40):
            for n in (2, 10, 33):
              cases += 1
              c = RotatingKVCache(max_size, keep)
              if patched: apply_rotating_sink_mask_patch(c)
              pos = 0
              while pos < history:
                  m = min(step, history - pos); step_chunk(c, m); pos += m
              mask, k = step_chunk(c, n)
              if not (mask is None or isinstance(mask, str)): engaged += 1
              P = history + n - 1
              got = visible(mask, k); want = expected(P, max_size, keep)
              # also: decode right after
              dm, dk = step_chunk(c, 1); dgot = visible(dm, dk); dwant = expected(P + 1, max_size, keep)
              if got != want or dgot != dwant:
                  bad.append((max_size, keep, step, history, n, sorted(got), sorted(want), sorted(dgot), sorted(dwant)))
    print(f"patched={patched}: cases {cases}, array masks {engaged}, mismatches {len(bad)} "
          f"(keep=0: {sum(1 for b in bad if b[1]==0)}, keep>0: {sum(1 for b in bad if b[1]>0)})")
    for b in bad[:6]: print("  ", b)
    return bad
u = run(False); p = run(True)
# the upstream PR reviewer's failing shape: max 8, keep 7, history 19, step 9, n 10
for patched in (False, True):
    c = RotatingKVCache(8, 7)
    if patched: apply_rotating_sink_mask_patch(c)
    for m in (9, 9, 1): step_chunk(c, m)
    mask, k = step_chunk(c, 10)
    print("reviewer case patched=%s last row sees %s, want %s" % (patched, sorted(visible(mask, k)), sorted(expected(28, 8, 7))))
