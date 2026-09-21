"""What the person at the terminal reads. Pure functions: status JSON in, text out.

Written for someone who runs an agent and has never read lmk's source: say what
is happening and what to type next, in words that need no glossary.
"""
import json
from pathlib import Path
from typing import Optional

FIRST_REQUEST_NOTE = (
    "The first request of a new conversation reads its whole prompt (about 3s per 1,000 tokens).\n"
    "  After that every step starts in about a second — also after a reboot.")


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


def _request_line(r: dict) -> str:
    who = " · ".join(x for x in (r.get("purpose"), r.get("ref_id")) if x)
    prefill = r.get("prefill")
    if r.get("phase") == "prefill" and prefill and prefill.get("total"):
        done, total = prefill["processed"], prefill["total"]
        doing = f"reading prompt {done:,} / {total:,} ({done * 100 // total}%)"
    elif r.get("phase") == "prefill":
        doing = "reading prompt"
    else:
        doing = "writing the answer"
    return " · ".join(x for x in (doing, who, human_duration(r.get("running_ms", 0))) if x)


def status_block(status: dict, url: str) -> str:
    model = status["model"]
    lines = [f"✓ lmk is up    {url}/v1   (OpenAI-compatible)",
             f"  model      {model['id']}   ({', '.join(model.get('input_modalities') or ['text'])} in)"]
    context = f"  context    {model['context_length']:,} tokens"
    requested = model.get("requested_context_length")
    if requested and model["context_length"] < requested:
        context += f"   (asked for {requested:,}; lowered to fit this Mac's memory)"
    lines.append(context)
    cache = status.get("cache")
    if cache:
        lines.append(f"  cache      {human_bytes(cache['used_bytes'])} of {human_bytes(cache['max_bytes'])}"
                     f"   {short_path(str(Path(cache['dir']).parent))}")
        if cache.get("disk_low"):
            lines.append("             the disk has less than 10 GB free — lmk has stopped adding to the cache")
    memory = status.get("memory")
    if memory:
        lines.append(f"  memory     pressure: {memory['pressure']} · {memory['free_percent']}% of "
                     f"{human_bytes(memory['total_bytes'])} free · lmk holds {human_bytes(memory['lmk_gpu_bytes'])}")
    lines.append(f"  running    {human_duration(status.get('uptime_ms', 0))}   (build {status.get('build', '?')})")
    in_flight = status.get("in_flight") or []
    if in_flight:
        lines.append(f"  busy       {len(in_flight)} request{'s' if len(in_flight) > 1 else ''}")
        lines.extend(f"               {_request_line(r)}" for r in in_flight)
    else:
        lines.append("  busy       no — idle")
    waiting = status.get("waiting") or []
    if waiting:
        lines.append(f"  waiting    {len(waiting)} request{'s' if len(waiting) > 1 else ''}")
        for w in waiting:
            who = " · ".join(x for x in (w.get("purpose"), w.get("ref_id")) if x) or "a request"
            lines.append(f"               {who} · {human_duration(w.get('waited_ms', 0))} · {w.get('reason')}")
    return "\n".join(lines)


def connect_block(url: str, model_id: str) -> str:
    return "\n".join([
        "  Point your agent at it — any OpenAI-compatible client:",
        f"    base URL   {url}/v1",
        f"    model      {model_id}",
        "    API key    anything (lmk does not check it)",
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
