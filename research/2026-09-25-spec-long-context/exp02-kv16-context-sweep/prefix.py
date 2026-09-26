"""Build the long-context prefixes from public source shipped with lmk's engine.

usage: prefix.py <model_dir> <out_dir> 8192 32768 ...
Concatenates, in sorted path order, the .py files of the mlx-engine fork, then mlx_lm, then mlx_vlm
(tests excluded), each under a "### <relative path>" header, and cuts the token sequence at each
target length. Writes <out_dir>/prefix-<n>.txt and prints the token count of each.
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / ".venv/lib/python3.11/site-packages"
SOURCES = [(ROOT / ".engine/mlx-engine", "mlx_engine"), (SITE, "mlx_lm"), (SITE, "mlx_vlm")]


def corpus(limit_chars: int) -> str:
    parts, size = [], 0
    for base, pkg in SOURCES:
        for f in sorted((base / pkg).rglob("*.py")):
            rel = f.relative_to(base).as_posix()
            if "/tests/" in rel or "__pycache__" in rel:
                continue
            text = f"### {rel}\n{f.read_text(encoding='utf-8', errors='replace')}\n"
            parts.append(text)
            size += len(text)
            if size >= limit_chars:
                return "".join(parts)
    return "".join(parts)


def main():
    model_dir, out_dir, sizes = sys.argv[1], Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok.encode(corpus(max(sizes) * 6), add_special_tokens=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        assert len(ids) >= n, (len(ids), n)
        text = tok.decode(ids[:n])
        (out_dir / f"prefix-{n}.txt").write_text(text, encoding="utf-8")
        print(n, len(tok.encode(text, add_special_tokens=False)), len(text))


if __name__ == "__main__":
    main()
