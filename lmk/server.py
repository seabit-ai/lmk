"""HTTP surface. OpenAI-shaped where a shape exists (design §5); lmk's own
endpoints live under /lmk/v1. State is answered on request — no subscriptions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lmk import log
from lmk.chat import CallerIdentity, ClientGone, prepare_messages, run_chat, run_warmup
from lmk.chatformat import ImageInputError
from lmk.clock import get_current_clock
from lmk.engine import Engine


class LmkServer:
    def __init__(self, engine: Engine, host: str, port: int, build: str = "dev", config_fingerprint: str = ""):
        self._engine = engine
        self._build = build
        self._config_fingerprint = config_fingerprint
        self._started_ms = get_current_clock().mono_ms()
        self._in_flight: dict[str, dict] = {}
        self._in_flight_lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # stdlib's access log → ours is structured
                pass

            def do_GET(self):
                outer._route_get(self)

            def do_POST(self):
                outer._route_post(self)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    # ---- routes ----

    def _route_get(self, h: BaseHTTPRequestHandler) -> None:
        if h.path == "/lmk/v1/status":
            _send_json(h, 200, self.status())
        elif h.path == "/v1/models":
            m = self._engine.loaded_model()
            _send_json(h, 200, {"object": "list", "data": [{"id": m.id, "object": "model", "owned_by": "lmk"}]})
        else:
            _send_json(h, 404, {"error": {"type": "not_found", "message": f"no route for GET {h.path}"}})

    def _route_post(self, h: BaseHTTPRequestHandler) -> None:
        if h.path not in ("/v1/chat/completions", "/lmk/v1/warmup"):
            _send_json(h, 404, {"error": {"type": "not_found", "message": f"no route for POST {h.path}"}})
            return
        try:
            body = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            _send_json(h, 400, {"error": {"type": "invalid_request", "message": "request body is not valid JSON"}})
            return
        resident = self._engine.loaded_model().id
        if body.get("model") != resident:
            # lmk serves exactly one resident model and never loads another on demand (design §6.7)
            _send_json(h, 404, {"error": {"type": "model_not_found", "param": "model",
                       "message": f"model {body.get('model')!r} is not served here; the resident model is {resident!r}"}})
            return
        try:
            prepare_messages(self._engine, body)
        except ImageInputError as e:
            _send_json(h, 400, {"error": {"type": "invalid_request", "param": "messages", "message": str(e)}})
            return
        identity = CallerIdentity(purpose=h.headers.get("X-Lmk-Purpose"), ref_id=h.headers.get("X-Lmk-Ref-Id"),
                                  traceparent=h.headers.get("traceparent"))
        if h.path == "/lmk/v1/warmup":
            self._track(h, identity, "warmup", lambda _progress: _send_json(h, 200, run_warmup(self._engine, body, identity)))
        else:
            self._chat(h, body, identity)

    def _track(self, h, identity: CallerIdentity, phase: str, work) -> None:
        """Run work while the call is visible in /lmk/v1/status."""
        key = str(id(h))
        entry = {"purpose": identity.purpose, "ref_id": identity.ref_id, "traceparent": identity.traceparent,
                 "started_mono_ms": get_current_clock().mono_ms(), "phase": phase, "prefill": None}
        with self._in_flight_lock:
            self._in_flight[key] = entry
        try:
            work(entry)
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(key, None)

    def _chat(self, h: BaseHTTPRequestHandler, body: dict, identity: CallerIdentity) -> None:
        self._track(h, identity, "prefill", lambda entry: self._chat_tracked(h, body, identity, entry))

    def _chat_tracked(self, h, body: dict, identity: CallerIdentity, entry: dict) -> None:
        def on_progress(progress: dict) -> None:
            entry["prefill"] = progress
            if progress["total"] and progress["processed"] >= progress["total"] - 1:
                entry["phase"] = "generating"

        if body.get("stream"):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Connection", "close")
            h.end_headers()

            def emit(chunk: dict) -> None:
                try:
                    h.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
                    h.wfile.flush()
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise ClientGone() from e

            result = run_chat(self._engine, body, identity, emit, on_progress)
            if not result["cancelled"]:
                try:
                    h.wfile.write(b"data: [DONE]\n\n")
                    h.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            h.close_connection = True
            return
        result = run_chat(self._engine, body, identity, lambda _chunk: None, on_progress)
        message = {"role": "assistant", "content": result["content"] or None}
        if result["reasoning_content"]:
            message["reasoning_content"] = result["reasoning_content"]
        if result["tool_calls"]:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in result["tool_calls"]]
        _send_json(h, 200, {**result["base"], "object": "chat.completion", "usage": result["usage"],
                            "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}]})

    def status(self) -> dict:
        m = self._engine.loaded_model()
        now = get_current_clock().mono_ms()
        with self._in_flight_lock:
            in_flight = [{"purpose": e["purpose"], "ref_id": e["ref_id"], "traceparent": e["traceparent"],
                          "phase": e["phase"], "prefill": e["prefill"], "running_ms": now - e["started_mono_ms"]}
                         for e in self._in_flight.values()]
        return {
            "build": self._build,
            "config_fingerprint": self._config_fingerprint,
            "model": {"id": m.id, "path": str(m.path), "context_length": m.context_length,
                      "requested_context_length": m.requested_context_length,
                      "input_modalities": self._engine.input_modalities()},
            "cache": self._engine.cache_stats(),
            "in_flight": in_flight,
            "uptime_ms": get_current_clock().mono_ms() - self._started_ms,
        }


def _send_json(h: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    raw = json.dumps(body, ensure_ascii=False).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    h.wfile.write(raw)
