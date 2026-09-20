"""HTTP surface. OpenAI-shaped where a shape exists (design §5); lmk's own
endpoints live under /lmk/v1. State is answered on request — no subscriptions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lmk import log
from lmk.clock import get_current_clock
from lmk.engine import Engine


class LmkServer:
    def __init__(self, engine: Engine, host: str, port: int):
        self._engine = engine
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

    def status(self) -> dict:
        m = self._engine.loaded_model()
        with self._in_flight_lock:
            in_flight = list(self._in_flight.values())
        return {
            "model": {"id": m.id, "path": str(m.path), "context_length": m.context_length},
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
