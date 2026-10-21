"""MTP draft heads for `model.repo` / `model.path` models (design 2026-10-20-draft-for-any-model).

Tested models name their draft in models.py. For any other model `model.draft_weights` says where
the head's weights come from: the model's own files (`model`, the default), lmk's copy of the
original Qwen3.8-27B head (`official`), or a draft folder / HuggingFace repo the user names.
Nothing here falls back to another source silently: every problem ends in a DraftProblem whose
message names the ways out."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

OWN, OFFICIAL = "model", "official"
CONVERTER_VERSION = 1  # bump when the conversion below changes: converted folders are keyed by it

# The one sidecar layout we have checked against the original head (research 2026-10-20, owner's notes):
# `mtp.`-prefixed tensors, 1-D norms stored zero-centred (the draft format stores them +1), quantized
# projections. Another layout may differ in exactly those details, so it is refused, not guessed.
_SIDECAR = Path("optiq") / "mtp.safetensors"
_SIDECAR_MARKER = Path("optiq") / "metadata.json"
# mlx-vlm's Qwen3.5 splitter adds 1.0 to these (mlx_vlm/speculative/drafters/qwen3_5_mtp/split.py)
_ZERO_CENTRED_NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight",
                       "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
_WAYS_OUT = ("set model.draft_weights to `official` (lmk's copy of the original Qwen3.8-27B head), to another draft's "
             "HuggingFace repo or folder (a folder starts with ./ or / or ~), or set speculative_decoding: false")


class DraftProblem(Exception):
    """A draft this config cannot use. The message is for the user; `lmk serve` exits cleanly on it."""


def is_folder(weights: str) -> bool:
    return weights.startswith(("./", "../", "/", "~"))


def own_head_draft(model_dir: Path, revision: Optional[str], drafts_dir: Path) -> Path:
    """A draft folder made from the head that ships with the model; converted once, then reused."""
    sidecar = model_dir / _SIDECAR
    if not sidecar.exists():
        if _head_in_main_weights(model_dir):
            raise DraftProblem("this model keeps its MTP head inside its main weight files, which lmk cannot split yet — "
                               + _WAYS_OUT)
        raise DraftProblem(f"this model has no MTP head lmk can read (looked for {_SIDECAR}) — " + _WAYS_OUT)
    if not (model_dir / _SIDECAR_MARKER).exists():
        raise DraftProblem(f"this model's {_SIDECAR} is in a layout lmk has not checked, so converting it could give a wrong head — "
                           + _WAYS_OUT)
    stat = sidecar.stat()
    source = revision or f"{model_dir.resolve()}|{stat.st_mtime_ns}"
    key = hashlib.sha256(f"{CONVERTER_VERSION}|{source}|{stat.st_size}".encode()).hexdigest()[:12]
    out = drafts_dir / f"mtp-{key}"
    if (out / "model.safetensors").exists():
        return out
    drafts_dir.mkdir(parents=True, exist_ok=True)
    tmp = drafts_dir / f".mtp-{key}.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        _convert(sidecar, model_dir, tmp)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _head_in_main_weights(model_dir: Path) -> bool:
    try:
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    except (OSError, ValueError, KeyError):
        return False
    return any(name.startswith("mtp.") for name in weight_map)


def _convert(sidecar: Path, model_dir: Path, out: Path) -> None:
    import mlx.core as mx  # not at import: the CLI paths must not load MLX

    model_cfg = json.loads((model_dir / "config.json").read_text())
    text = dict(model_cfg.get("text_config") or model_cfg)
    group_size = int((model_cfg.get("quantization") or {}).get("group_size", 64))
    tensors = mx.load(str(sidecar))
    if not all(name.startswith("mtp.") for name in tensors):
        raise DraftProblem(f"{_SIDECAR} holds tensors that are not the MTP head's (all names should start with mtp.) — " + _WAYS_OUT)
    converted, quantization = {}, {"group_size": group_size, "bits": 4, "mode": "affine"}
    for name, value in tensors.items():
        name = name[len("mtp."):]
        if value.ndim == 1 and name.endswith(_ZERO_CENTRED_NORMS):
            value = value + 1.0
        converted[name] = value
        if name.endswith(".scales"):
            module = name[:-len(".scales")]
            in_features = value.shape[1] * group_size
            packed_columns = tensors["mtp." + module + ".weight"].shape[1]  # in_features weights packed into 32-bit words
            quantization[module] = {"bits": packed_columns * 32 // in_features, "group_size": group_size}
    mx.save_safetensors(str(out / "model.safetensors"), converted, metadata={"format": "mlx"})
    rope = text.get("rope_parameters")
    if isinstance(rope, dict) and "type" in rope and "rope_type" not in rope:  # the draft format's spelling
        text["rope_parameters"] = {("rope_type" if k == "type" else k): v for k, v in rope.items()}
    depth = int(text.get("mtp_num_hidden_layers") or 1)
    config = {"block_size": depth + 2, "model_type": "qwen3_5_mtp", "text_config": text,  # mlx-vlm's splitter: depth + 2
              "tie_word_embeddings": False, "quantization": quantization}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "lmk-draft.json").write_text(json.dumps({"converter": CONVERTER_VERSION, "from": str(sidecar)}))


def resolve(source, weights: Optional[str], model_dir: Path, revision: Optional[str], drafts_dir: Path) -> Path:
    """The MTP draft folder for a repo / path model, from model.draft_weights (None = the model's own).
    Raises DraftProblem, or models.DraftNotDownloaded for a repo that `lmk up` has yet to fetch."""
    from lmk.models import resolve_draft  # models imports this module for is_folder

    weights = weights or OWN
    if weights == OWN:
        folder = own_head_draft(model_dir, revision, drafts_dir)
    elif is_folder(weights):
        folder = Path(weights).expanduser()
        if not folder.is_dir():
            raise DraftProblem(f"model.draft_weights {weights} is not a folder on this Mac")
    else:
        folder = resolve_draft(source, "mtp", weights)
    problem = incompatible(folder, model_dir)
    if problem:
        raise DraftProblem(problem + " — " + _WAYS_OUT)
    return folder


def incompatible(draft_dir: Path, model_dir: Path) -> Optional[str]:
    """Why this draft cannot serve this model, or None. Reads two config.json files."""
    try:
        draft = json.loads((draft_dir / "config.json").read_text())
    except (OSError, ValueError) as e:
        return f"{draft_dir} has no readable config.json ({e}); a draft folder holds config.json and model.safetensors"
    model = json.loads((model_dir / "config.json").read_text())
    if "dflash_config" in draft or any("DFlash" in a for a in draft.get("architectures") or []):
        return f"{draft_dir} is a DFlash drafter; for this model only MTP heads can be used"
    drafted, target = draft.get("text_config") or draft, model.get("text_config") or model
    for key in ("hidden_size", "vocab_size"):
        if drafted.get(key) is not None and target.get(key) is not None and drafted[key] != target[key]:
            return f"the draft's {key} is {drafted[key]} but the model's is {target[key]}, so it was made for a different model"
    return None
