"""exp01: cancel a prefill halfway by closing the stream, then send the same request again
and see how much of it is still cached. Stdlib only; talks to the running lmk service."""
import http.client
import json
import sys
import time
import uuid

HOST, PORT = "127.0.0.1", 1235
MODEL = "qwen3.8-27b-4bit"
TARGET_TOKENS_APPROX = 16000
CANCEL_AT = 0.5

PARAGRAPH = ("The warehouse inventory report lists each shelf, the items stored on it, the date they "
             "arrived, and the person who last counted them. Counts are checked against the delivery "
             "notes every Friday, and any difference is written down with a short explanation. ")


def build_body() -> dict:
    nonce = uuid.uuid4().hex
    parts = [f"Experiment nonce: {nonce}."]
    i = 0
    # ~60 tokens per paragraph line; stop near the target
    while len(parts) * 60 < TARGET_TOKENS_APPROX:
        i += 1
        parts.append(f"Section {i}. {PARAGRAPH}")
    return {"model": MODEL, "max_tokens": 1,
            "messages": [{"role": "system", "content": "\n".join(parts)},
                         {"role": "user", "content": "Reply with one word: ok."}]}


def status() -> dict:
    c = http.client.HTTPConnection(HOST, PORT, timeout=10)
    c.request("GET", "/lmk/v1/status")
    s = json.loads(c.getresponse().read())
    c.close()
    return s


def log(**kw) -> None:
    kw["t_ms"] = int(time.monotonic() * 1000)
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def run_a(body: dict) -> None:
    b = dict(body, stream=True)
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(b),
              {"Content-Type": "application/json", "X-Lmk-Purpose": "research", "X-Lmk-Ref-Id": "prewarm-exp01-A"})
    r = c.getresponse()
    log(step="A", event="headers", status=r.status)
    last = None
    while True:
        line = r.fp.readline()
        if not line:
            log(step="A", event="stream-ended-before-cancel", last=last)
            break
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        p = (chunk.get("lmk") or {}).get("prefill")
        if p:
            last = p
            log(step="A", event="prefill", **p, elapsed_ms=int((time.monotonic() - t0) * 1000))
            if p["total"] and p["processed"] / p["total"] >= CANCEL_AT:
                log(step="A", event="cancel", **p, elapsed_ms=int((time.monotonic() - t0) * 1000))
                break
    r.close()
    c.close()


def wait_idle() -> None:
    t0 = time.monotonic()
    while True:
        s = status()
        if not s["in_flight"] and not s["waiting"]:
            log(step="wait", event="idle", waited_ms=int((time.monotonic() - t0) * 1000), recent=s.get("recent", [])[:1])
            return
        time.sleep(0.5)


def run_b(body: dict) -> None:
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json", "X-Lmk-Purpose": "research", "X-Lmk-Ref-Id": "prewarm-exp01-B"})
    resp = json.loads(c.getresponse().read())
    c.close()
    u = resp.get("usage", {})
    log(step="B", event="done", prompt_tokens=u.get("prompt_tokens"),
        cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        elapsed_ms=int((time.monotonic() - t0) * 1000))


def main() -> None:
    s = status()
    if s["in_flight"] or s["waiting"]:
        log(event="abort-busy", in_flight=s["in_flight"], waiting=s["waiting"])
        sys.exit(1)
    body = build_body()
    log(event="start", model=MODEL, system_chars=len(body["messages"][0]["content"]), cancel_at=CANCEL_AT)
    run_a(body)
    wait_idle()
    run_b(body)


if __name__ == "__main__":
    main()
