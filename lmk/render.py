"""What the person at the terminal reads. Pure functions: status JSON in, text out.

Written for someone who runs an agent and has never read lmk's source: say what
is happening and what to type next, in words that need no glossary.
"""
import json
from pathlib import Path
from typing import Optional

from lmk.models import TESTED_MODELS, model_page_url
from lmk.sampling import describe as describe_sampling

FIRST_REQUEST_NOTE = (
    "The first request of a new conversation reads its whole prompt (about 3s per 1,000 tokens).\n"
    "  After that every step starts in about a second — also after a reboot.")


def loading_line(model_id: str, resident_bytes: Optional[int], total_bytes: int) -> str:
    """Progress by bytes in memory against the model's weight bytes — never by seconds, which say
    nothing about how far along a load is. Resident overshoots the weights by the runtime's own
    ~0.4 GB, so it is capped at the total."""
    if not total_bytes or resident_bytes is None:
        return f"loading {model_id} …"
    value, unit = human_bytes(total_bytes).split()  # both numbers in the total's unit: "8.2 of 16.0 GB"
    scale = total_bytes / float(value) if float(value) else 1
    loaded = min(resident_bytes, total_bytes) / scale
    return f"loading {model_id} … {loaded:.{len(value.partition('.')[2])}f} of {value} {unit}"


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def human_duration(ms: int) -> str:
    s = ms // 1000
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {s % 3600 // 60}m"
    return f"{s // 86400}d {s % 86400 // 3600}h"


def short_path(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


def base_url(host: str, port: int) -> str:
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"


UNNAMED = "(unnamed)"
UNNAMED_NOTE = ("  (unnamed): the client did not say who it is. It can, with two optional request headers —\n"
                "  X-Lmk-Purpose (turn, groom, …) and X-Lmk-Ref-Id (its own reference for the call).")


def _who(r: dict) -> str:
    return " · ".join(x for x in (r.get("purpose"), r.get("ref_id")) if x) or UNNAMED


def _prompt(r: dict) -> str:
    cached = r.get("cached_tokens")
    return f"{r.get('prompt_tokens', 0):,} prompt" + (f" ({cached:,} cached)" if cached is not None else "")


def _rate(r: dict) -> str:
    rate = r.get("decode_tokens_per_s")
    return f"{r.get('completion_tokens', 0):,} tokens" + (f" at {rate:.0f}/s" if rate else "")


def _request_rows(status: dict) -> list[tuple[str, str, str, str]]:
    """(state, detail, who, what) for every request lmk holds: running first, then the queue."""
    rows = []
    for r in status.get("in_flight") or []:
        state, took = r.get("state") or "starting", human_duration(r.get("running_ms", 0))
        prefill = r.get("prefill")
        if state == "prefill" and prefill and prefill.get("total"):
            done, total = prefill["processed"], prefill["total"]
            rows.append(("prefill", f"{done * 100 // total}%", _who(r),
                         f"{done:,} / {total:,} · {prefill.get('cached', 0):,} cached · {took}"))
        elif state == "decode":
            rows.append(("decode", r.get("part") or "", _who(r), f"{_prompt(r)} · {_rate(r)} · {took}"))
        else:
            rows.append((state, "", _who(r), f"{_prompt(r)} · {took}"))
    for w in status.get("waiting") or []:
        rows.append(("queued", "", _who(w), f"{human_duration(w.get('waited_ms', 0))} · {w.get('reason')}"))
    return rows


def _finished_line(f: dict, who_width: int) -> str:
    first = f.get("first_token_ms")
    parts = [_prompt(f)]
    if first is not None:
        parts.append(f"first token {first / 1000:.1f}s")
    if f.get("completion_tokens"):
        parts.append(_rate(f))
    parts += [f.get("outcome") or "?", f"{human_duration(f.get('ago_ms', 0))} ago"]
    return f"  {_who(f):<{who_width}}  " + " · ".join(parts)


def status_block(status: dict, url: str) -> str:
    model = status["model"]
    context = f"{model['context_length']:,} tokens"
    requested = model.get("requested_context_length")
    if requested and model["context_length"] < requested:
        context += f" (asked for {requested:,}; lowered to fit this Mac's memory)"
    lines = [f"✓ lmk is up    {url}/v1   (OpenAI-compatible)",
             f"  model      {model['id']} · {', '.join(model.get('input_modalities') or ['text'])} in · {context}"
             + (f" · thinking {'on' if model['thinking'] else 'off'}" if "thinking" in model else "")
             + (f" · KV cache {model['kv_cache_bits']}-bit" if model.get("kv_cache_bits", 16) != 16 else "")
             + (" · speculative decoding on" if model.get("speculative_decoding") else "")]
    draft = status.get("draft")
    if draft and draft.get("drafted"):
        rate = draft["accepted"] / draft["drafted"]
        per_round = (draft["accepted"] + draft["rounds"]) / max(1, draft["rounds"])
        lines.append(f"  draft      {rate:.0%} of drafted tokens accepted ({draft['accepted']:,} of {draft['drafted']:,}) · "
                     f"{per_round:.1f} tokens per round")
    if model["id"] in TESTED_MODELS:
        lines.append(f"  about it   {model_page_url(model['id'])}")
    if "sampling_defaults" in status:
        lines.append(f"  sampling   {describe_sampling(status['sampling_defaults'])}   "
                     "(the model's generation_config; a request may override)")
    memory = status.get("memory")
    if memory:
        lines.append(f"  memory     pressure: {memory['pressure']} · {memory['free_percent']}% of "
                     f"{human_bytes(memory['total_bytes'])} free · lmk holds {human_bytes(memory['lmk_gpu_bytes'])}")
    cache, totals = status.get("cache"), status.get("totals") or {}
    if cache:
        line = (f"  cache      {human_bytes(cache['used_bytes'])} of {human_bytes(cache['max_bytes'])} in "
                f"{short_path(str(Path(cache['dir']).parent))}")
        if totals.get("prompt_tokens"):
            # a ratio, with both of its numbers: the size of the cache alone does not say whether it works
            line += (f" · since start {totals['cached_tokens'] * 100 // totals['prompt_tokens']}% of prompt tokens "
                     f"came from it ({totals['cached_tokens']:,} of {totals['prompt_tokens']:,})")
        lines.append(line)
        if cache.get("disk_low"):
            lines.append("             the disk has less than 10 GB free — lmk has stopped adding to the cache")
    requests = status.get("requests")
    if requests:
        line = (f"  requests   answering {requests['answering']} of {requests['max_parallel']} · "
                f"waiting {requests['waiting']} of {requests['max_queue']}")
        if requests.get("token_budget"):
            line += f" · tokens in memory {requests['tokens_in_memory']:,} of {requests['token_budget']:,}"
        lines.append(line)
    since = [f"{totals[k]:,} {k}" for k in ("answered", "refused", "failed", "cancelled") if k in totals]
    lines.append("  since start  " + " · ".join(since + [f"up {human_duration(status.get('uptime_ms', 0))}",
                                                           f"build {status.get('build', '?')}"]))
    lines.append("")
    rows = _request_rows(status)
    if rows:
        who_width = min(40, max(len(r[2]) for r in rows))
        lines += [f"  {state:<8} {detail:<10} {who:<{who_width}}  {what}" for state, detail, who, what in rows]
    else:
        lines.append("  idle — no requests")
    recent = status.get("recent") or []
    if recent:
        who_width = min(40, max(len(_who(f)) for f in recent))
        lines += ["", "  just finished"] + [_finished_line(f, who_width) for f in recent]
    everyone = (status.get("in_flight") or []) + (status.get("waiting") or []) + recent
    if any(_who(r) == UNNAMED for r in everyone):
        lines += ["", UNNAMED_NOTE]
    return "\n".join(lines)


# One observed agent step wrote 15,698 tokens, thinking included (research/2026-09-20-thinking-length, THK-004);
# the 8192 in OpenClaw's own local-model template would have cut it off.
OPENCLAW_MAX_TOKENS = 32768


def connect_block(url: str, model: dict) -> str:
    """`model` is the `model` object of /lmk/v1/status."""
    model_id = model["id"]
    modalities = ", ".join(f'"{m}"' for m in (model.get("input_modalities") or ["text"]))
    return "\n".join([
        "  Point your agent at it — any OpenAI-compatible client:",
        f"    base URL   {url}/v1",
        f"    model      {model_id}",
        "    API key    anything (lmk does not check it)",
        "",
        "  OpenClaw — merge this into ~/.openclaw/openclaw.json, then pick the model "
        f"lmk/{model_id}:",
        "    models: {",
        '      mode: "merge",',
        "      providers: {",
        "        lmk: {",
        f'          baseUrl: "{url}/v1",',
        '          apiKey: "lmk",',
        '          api: "openai-completions",',
        "          models: [{",
        f'            id: "{model_id}",',
        f'            name: "{model_id} (lmk)",',
        "            reasoning: false,",
        f"            input: [{modalities}],",
        "            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },",
        f"            contextWindow: {model['context_length']},",
        f"            maxTokens: {OPENCLAW_MAX_TOKENS},",
        "          }],",
        "        },",
        "      },",
        "    },",
        "",
        "  kitten — put this in .kitten/config.yaml:",
        "    providers:",
        "      lmk:",
        f"        base_url: {url}",
        "    model_aliases:",
        f'      local: "lmk:{model_id}"',
        "",
        f"  {FIRST_REQUEST_NOTE}",
        "  It starts by itself at login. Check on it: lmk status   Stop it: lmk down",
    ])


def not_downloaded_block(repo: str, why: str, size_gb: Optional[float]) -> str:
    size = f" ({size_gb:.0f} GB)" if size_gb else ""
    return f"✗ model {why}: {repo}{size}\n  run:  lmk pull"


def log_line(raw: str) -> str:
    """One line of lmk.jsonl → `HH:MM:SS LEVEL Event  message  key=value …`"""
    import datetime

    try:
        rec = json.loads(raw)
    except ValueError:
        return raw.rstrip("\n")
    when = datetime.datetime.fromtimestamp(rec.pop("time_ms", 0) / 1000).strftime("%H:%M:%S")
    level, event, msg = rec.pop("level", "INFO"), rec.pop("event", "?"), rec.pop("msg", "")
    rest = "  ".join(f"{k}={v}" for k, v in rec.items())
    return f"{when} {level:<5} {event}  {msg}" + (f"  {rest}" if rest else "")
