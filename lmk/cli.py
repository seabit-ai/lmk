"""lmk pull / up / status / logs / down  (design OOBE §A, §E).

One machine, one model, always on. Downloading is its own explicit command;
`up` returns only when the server answers a real request.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from lmk import render, service
from lmk.clock import get_current_clock
from lmk.config import ConfigError, LmkConfig, app_dir, config_path, fingerprint, load_config
from lmk.configfiles import refresh_example, seed_config
from lmk.models import TESTED_MODELS, ModelNotDownloaded, missing_weight_files, resolve_model

READY_TIMEOUT_S = 600  # a cold load of a large model from a slow disk; normally ~10s


def _say(text: str = "") -> None:
    print(text, flush=True)


def _fail(text: str, code: int) -> int:
    print(text, file=sys.stderr, flush=True)
    return code


def _config() -> LmkConfig:
    if seed_config(config_path()):
        _say(f"  wrote {render.short_path(str(config_path()))} (all comments — lmk runs fine without touching it)")
    refresh_example(config_path())
    return load_config()


def _get_status(cfg: LmkConfig, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{render.base_url(cfg.host, cfg.port)}/lmk/v1/status", timeout=timeout) as r:
            body = json.loads(r.read())
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) and "model" in body and "in_flight" in body else None


def _tail(path: Path, lines: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            return f.read().decode(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


# ---- pull ----

def cmd_pull(_args) -> int:
    cfg = _config()
    source = cfg.model.source
    if source.kind == "path":
        _say(f"model.path points at a directory on this disk ({source.path}) — there is nothing to download.")
        return 0
    try:
        resolve_model(source)
        _say(f"✓ {source.repo} is already downloaded.\n  next:  lmk up")
        return 0
    except ModelNotDownloaded:
        pass

    # mlx-engine, once imported, replaces snapshot_download with a function that always
    # raises. Nothing on the `lmk pull` path imports the engine; keep it that way.
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    try:
        info = HfApi().model_info(source.repo, files_metadata=True)
    except Exception as e:  # network, auth, a repo that does not exist: all end the same way for the user
        return _fail(f"✗ cannot reach {source.repo} on HuggingFace: {e}", 1)
    total = sum(s.size or 0 for s in info.siblings)
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(HF_HUB_CACHE).free
    if free < total * 1.05:
        return _fail(f"✗ not enough disk space: {source.repo} is {render.human_bytes(total)}, "
                     f"{render.human_bytes(free)} free in {render.short_path(str(HF_HUB_CACHE))}", 1)
    _say(f"Downloading {source.repo} — {render.human_bytes(total)} into {render.short_path(str(HF_HUB_CACHE))}\n"
         "(the shared HuggingFace cache; interrupt any time, `lmk pull` picks up where it stopped)\n")
    try:
        snapshot = Path(snapshot_download(source.repo))
    except KeyboardInterrupt:
        return _fail("\nstopped. `lmk pull` continues from here.", 130)
    except Exception as e:
        return _fail(f"\n✗ download failed: {e}\n  `lmk pull` continues from where it stopped.", 1)
    missing = missing_weight_files(snapshot)
    if missing:
        return _fail(f"✗ the download finished but {len(missing)} weight file(s) are missing — run `lmk pull` again", 1)
    _say(f"\n✓ downloaded {source.repo}\n  next:  lmk up")
    return 0


# ---- up ----

def _why_it_did_not_start(cfg: LmkConfig) -> str:
    out = []
    events = [render.log_line(l) for l in _tail(cfg.log_dir / "lmk.jsonl", 8)]
    if events:
        out += [f"  last events ({render.short_path(str(cfg.log_dir / 'lmk.jsonl'))}):"] + [f"    {l}" for l in events]
    stderr = _tail(cfg.log_dir / "lmk.stderr.log", 15)
    if stderr:
        out += [f"  last output ({render.short_path(str(cfg.log_dir / 'lmk.stderr.log'))}):"] + [f"    {l}" for l in stderr]
    return "\n".join(out)


def _smoke(cfg: LmkConfig) -> Optional[str]:
    body = json.dumps({"model": cfg.model.id, "max_tokens": 8, "stream": False,
                       "messages": [{"role": "user", "content": "Say OK."}]}).encode()
    req = urllib.request.Request(f"{render.base_url(cfg.host, cfg.port)}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "smoke"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            json.loads(r.read())["choices"][0]
    except (OSError, ValueError, KeyError, IndexError) as e:
        return str(e)
    return None


def cmd_up(_args) -> int:
    clock = get_current_clock()
    cfg = _config()
    if (app_dir() / ".git").exists():
        return _fail("✗ this is a source checkout. The service must not run from a working tree (switching\n"
                     "  branches would take it down) — install it first:  make install", 2)
    try:
        resolved = resolve_model(cfg.model.source)
    except ModelNotDownloaded as e:
        tested = TESTED_MODELS.get(cfg.model.source.value) if cfg.model.source.kind == "name" else None
        return _fail(render.not_downloaded_block(e.repo, e.why, tested.size_gb if tested else None), 3)
    except FileNotFoundError as e:
        return _fail(f"✗ {e}", 3)

    from lmk.modelfit import why_it_does_not_fit

    too_big = why_it_does_not_fit(resolved.path)
    if too_big:
        return _fail(f"✗ {too_big}", 3)

    url = render.base_url(cfg.host, cfg.port)
    running = _get_status(cfg)
    if running is not None and running.get("config_fingerprint") == fingerprint(cfg, resolved.revision):
        _say(render.status_block(running, url) + "\n\n" + render.connect_block(url, running["model"]))
        return 0
    if running is None:
        owner = service.port_owner(cfg.port)
        if owner and not service.is_registered():
            return _fail(f"✗ port {cfg.port} is taken by {owner}.\n  Stop that, or pick another port under "
                         f"listen: in {render.short_path(str(config_path()))}", 4)

    if running is not None or service.is_registered():
        _say("Restarting lmk — the code or the configuration changed since it started." if running is not None
             else "lmk is registered but not answering; starting it again.")
        if running is not None and running.get("in_flight"):
            _say(f"  ({len(running['in_flight'])} request(s) in progress will be cut off)")
        from lmk.persistcache import model_identity

        was = Path(((running or {}).get("cache") or {}).get("dir") or "").name
        now = model_identity(resolved.path, repo=cfg.model.source.repo, revision=resolved.revision)
        if was and was != now and not (cfg.cache_dir / now).is_dir():
            _say("  The prompt cache starts empty for this model: the first request of each conversation\n"
                 "  reads its whole prompt again (about 3s per 1,000 tokens).")
        if not service.stop(clock.sleep_s):
            return _fail("✗ the running lmk did not stop within two minutes — see `lmk logs`", 5)

    service.start(app_dir(), cfg.log_dir, dict(os.environ))
    started = clock.mono_ms()
    live = sys.stdout.isatty()  # a counter that rewrites its line is noise in a pipe or a log
    if not live:
        _say(f"  loading {cfg.model.id} …")
    while True:
        status = _get_status(cfg)
        if status is not None:
            break
        waited = (clock.mono_ms() - started) // 1000
        if not service.is_registered():
            return _fail("✗ lmk exited while starting.\n" + _why_it_did_not_start(cfg), 5)
        if waited > READY_TIMEOUT_S:
            return _fail(f"✗ lmk did not answer within {READY_TIMEOUT_S}s.\n" + _why_it_did_not_start(cfg), 5)
        if live:
            print(f"\r  loading {cfg.model.id} … {waited}s", end="", flush=True)
        clock.sleep_s(1)
    if live:
        print("\r" + " " * 60 + "\r", end="")

    problem = _smoke(cfg)
    if problem:
        return _fail(f"✗ lmk started but a test request failed: {problem}\n" + _why_it_did_not_start(cfg), 5)
    status = _get_status(cfg) or status
    _say(render.status_block(status, url) + "\n\n" + render.connect_block(url, status["model"]))
    return 0


# ---- status / logs / down ----

def _status_text(cfg: LmkConfig) -> tuple[str, bool]:
    status = _get_status(cfg)
    if status is not None:
        return render.status_block(status, render.base_url(cfg.host, cfg.port)), True
    if service.is_registered():
        return ("… lmk is starting (or failing to) — it is registered but not answering yet.\n"
                "  watch it:  lmk logs -f"), False
    return "✗ lmk is not running.\n  start it:  lmk up", False


def cmd_status(args) -> int:
    cfg = _config()
    if args.json:
        status = _get_status(cfg)
        _say(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if status else 1
    if not args.watch:
        text, up = _status_text(cfg)
        _say(text)
        return 0 if up else 1
    clock = get_current_clock()
    try:
        while True:
            text, _ = _status_text(cfg)
            # home + clear-to-end, then the text: no flicker, and a shorter screen leaves nothing behind
            print("\033[H\033[J" + text + "\n\n  refreshing every second — ctrl-c to leave", flush=True)
            clock.sleep_s(1)
    except KeyboardInterrupt:
        return 0


def cmd_logs(args) -> int:
    cfg = _config()
    path = cfg.log_dir / ("lmk.stderr.log" if args.raw else "lmk.jsonl")
    show = (lambda l: l.rstrip("\n")) if args.raw else render.log_line
    if not path.exists():
        return _fail(f"no log yet at {path}", 1)
    for line in _tail(path, args.lines):
        _say(show(line))
    if not args.follow:
        return 0
    clock = get_current_clock()
    with open(path, "r", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                line = f.readline()
                if line:
                    _say(show(line))
                else:
                    clock.sleep_s(0.5)
        except KeyboardInterrupt:
            return 0


def cmd_down(_args) -> int:
    if not service.is_registered() and not service.plist_path().exists():
        _say("lmk is not running, and not set to start at login. Nothing to do.")
        return 0
    if not service.stop(get_current_clock().sleep_s):
        return _fail("✗ lmk did not stop within two minutes — see `lmk logs`", 5)
    service.remove_plist()
    _say("✓ lmk is stopped and will not start at login.\n"
         "  Your model, prompt cache and configuration are kept.  Start again:  lmk up")
    return 0


def cmd_bench(args) -> int:
    import datetime

    from lmk import bench

    if args.url:
        url = args.url.rstrip("/")
        try:
            with urllib.request.urlopen(f"{url}/lmk/v1/status", timeout=2) as r:
                status = json.loads(r.read())
        except (OSError, ValueError):
            return _fail(f"✗ no lmk answers at {url}", 1)
    else:
        cfg = load_config()
        url = render.base_url(cfg.host, cfg.port)
        status = _get_status(cfg)
        if status is None:
            return _fail("✗ lmk is not running.  Start it:  lmk up", 1)
    if status.get("in_flight") or status.get("waiting"):
        return _fail("✗ lmk is busy; a benchmark needs it to itself.  See:  lmk status", 1)
    model = status["model"]
    m = bench.machine()
    _say(f"{model['id']} on {m['chip']} {m['memory_gb']} GB — measuring, about a minute")
    result = bench.run_bench(bench.stream_via_http(url), model["id"], seed=args.seed)
    _say(bench.human_block(result))
    _say("")
    _say(f"  seed {result.seed}   ·   row for docs/benchmarks.md:")
    _say(bench.markdown_row(result, m, status, datetime.date.today().isoformat()))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lmk", description="One local model, always on, for your agent.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser("pull", help="download the configured model (explicit; nothing else ever downloads)")
    sub.add_parser("up", help="start lmk now and at every login; returns when it answers")
    p = sub.add_parser("status", help="is it up, what each request is doing, memory, cache, the last few answers")
    p.add_argument("--json", action="store_true", help="the raw status document")
    p.add_argument("-w", "--watch", action="store_true", help="keep the screen up to date (macOS has no `watch`)")
    p = sub.add_parser("logs", help="recent events")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("--raw", action="store_true", help="the model runtime's own output instead of lmk's events")
    sub.add_parser("down", help="stop lmk and do not start it at login")
    p = sub.add_parser("bench", help="measure prefill, cache-hit and decode speed on this Mac; prints a row for docs/benchmarks.md")
    p.add_argument("--url", help="an lmk other than the configured one (default: this Mac's)")
    p.add_argument("--seed", type=int, help="reuse a seed from an earlier run: the cold probe then tests whether "
                                            "the cache still holds that prompt (default: a fresh one, so it is cold)")
    sub.add_parser("serve", help=argparse.SUPPRESS)  # what launchd runs

    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from lmk.serve import serve

            return serve()
        return {"pull": cmd_pull, "up": cmd_up, "status": cmd_status, "logs": cmd_logs, "down": cmd_down,
                "bench": cmd_bench}[args.command](args)
    except ConfigError as e:
        return _fail(f"✗ {render.short_path(str(config_path()))}: {e}", 2)
