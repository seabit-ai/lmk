"""`lmk report`: everything a maintainer needs to see what happened on this Mac, as one Markdown
block for a GitHub issue. lmk cannot tell why a Mac misbehaves (only report numbers, never guess
causes); it can make the whole scene travel: machine, build, configuration, status, the canary
answer, the last events and the last traceback. Nothing is sent anywhere — the user pastes it.
"""
from typing import Optional

from lmk.bench import CANARY_EXPECTED, Probe, canary_matches
from lmk.render import status_block

ISSUES_URL = "https://github.com/seabit-ai/lmk/issues/new"


def markdown(*, machine: dict, build: str, engine: str, config_text: str, config_path: str, status: Optional[dict],
             canary: Optional[Probe], canary_error: Optional[str], events: list[str], stderr: list[str],
             bench_row: Optional[str] = None, url: str = "") -> str:
    out = [f"<!-- lmk report — paste into {ISSUES_URL} . Review it first: it holds your config and the last log lines. -->",
           "## Machine",
           f"- {machine.get('chip', '?')} · {machine.get('gpu_cores', '?')} GPU cores · {machine.get('memory_gb', '?')} GB · "
           f"{machine.get('model_identifier', '?')} · macOS {machine.get('macos', '?')}",
           f"- lmk {build} · engine {engine}",
           "", f"## Configuration (`{config_path}`)", "```yaml", config_text.rstrip("\n"), "```", ""]
    out.append("## Canary (a fixed prompt, temperature 0; the answer must contain 1, 2, … 20)")
    if canary is not None:
        verdict = "matches the reference" if canary_matches(canary.text) else "**DIFFERS from the reference — this Mac may be producing wrong text**"
        out += [f"- {verdict}", f"- answer ({canary.completion_tokens} tokens, first token {canary.first_token_ms / 1000:.2f} s):",
                "```", canary.text.strip()[:600], "```"]
    else:
        out.append(f"- not run: {canary_error or 'lmk is not running'}")
    out.append("")
    if bench_row:
        out += ["## Bench", bench_row, ""]
    out.append("## Status (`lmk status`)")
    if status is None:
        out.append("- lmk is not running (or did not answer)")
    else:
        out += ["```", status_block(status, url), "```"]
    out += ["", "## Last events (`lmk logs`)", "```"] + [l.rstrip("\n") for l in events] + ["```", ""]
    # the engine's own lines and tracebacks; the JSON event lines also land in stderr and are above already
    raw = [l.rstrip("\n") for l in stderr if not l.startswith("{\"time_ms\"")]
    out += ["## Last output (`lmk logs --raw`: the engine's lines and tracebacks)", "```"] + raw + ["```"]
    return "\n".join(out) + "\n"


__all__ = ["markdown", "ISSUES_URL", "CANARY_EXPECTED"]
