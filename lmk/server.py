"""HTTP surface. OpenAI-shaped where a shape exists (design §5); lmk's own
endpoints live under /lmk/v1. State is answered on request — no subscriptions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lmk import log
from lmk.admission import Admission, QueueFull, Ticket, WaitedTooLong
from lmk.board import Board, Running
from lmk.chat import CallerIdentity, ClientGone, prepare_chat, run_chat, run_warmup
from lmk.chatformat import ImageInputError
from lmk.clock import get_current_clock
from lmk.config import RequestsConfig
from lmk.engine import Engine, engine_commit
from lmk.sampling import SamplingError
from lmk.memory import get_current_memory
from lmk.models import mac_memory_gb


class LmkServer:
    def __init__(self, engine: Engine, host: str, port: int, build: str = "dev", config_fingerprint: str = "",
                 requests: RequestsConfig = RequestsConfig(2, 16, 600), admission: Admission = None,
                 kv_cache_bits_auto: bool = False):
        self._engine = engine
        self._kv_cache_bits_auto = kv_cache_bits_auto
        self._admission = admission or Admission(requests.max_parallel, requests.max_queue,
                                                 requests.max_wait_seconds, token_budget=engine.token_budget())
        self._build = build
        self._config_fingerprint = config_fingerprint
        self._started_ms = get_current_clock().mono_ms()
        self._board = Board()
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
        warmup = h.path == "/lmk/v1/warmup"
        try:
            prepared = prepare_chat(self._engine, body, warmup=warmup)
        except ImageInputError as e:
            _send_json(h, 400, {"error": {"type": "invalid_request", "param": "messages", "message": str(e)}})
            return
        except SamplingError as e:
            _send_json(h, 400, {"error": {"type": "invalid_request", "param": e.param, "message": str(e)}})
            return
        identity = CallerIdentity(purpose=h.headers.get("X-Lmk-Purpose"), ref_id=h.headers.get("X-Lmk-Ref-Id"),
                                  traceparent=h.headers.get("traceparent"))
        ticket = Ticket(purpose=identity.purpose, ref_id=identity.ref_id,
                        tokens=prepared.tokens_needed(self._engine.loaded_model().context_length),
                        uncached_tokens=prepared.preflight.uncached_tokens, background=warmup)
        # Nothing has been sent yet, and nothing is until the queue lets the request in:
        # one that cannot start gets a plain 503 with the reason, not a broken stream.
        try:
            self._admission.enter(ticket)
        except (QueueFull, WaitedTooLong) as e:
            if warmup:
                # a warmup that cannot start is not an error: its caller asks again later
                log.info("LmkWarmupNotStarted", "warmup gave up waiting for an idle engine",
                         purpose=identity.purpose, refId=identity.ref_id, reason=str(e))
                _send_json(h, 200, {"outcome": "not_started", "reason": getattr(e, "reason", str(e))})
                return
            self._board.refused()
            if isinstance(e, QueueFull):
                log.warn("LmkQueueFull", "request refused: the queue is full", purpose=identity.purpose,
                         refId=identity.ref_id, waiting=e.waiting)
                _send_json(h, 503, {"error": {"type": "queue_full", "message": f"lmk is busy: {e}. Try again shortly."}})
            else:
                log.warn("LmkWaitedTooLong", "request refused: it could not start in time", purpose=identity.purpose,
                         refId=identity.ref_id, waitedS=e.waited_s, reason=e.reason)
                _send_json(h, 503, {"error": {"type": "waited_too_long", "message": f"lmk {e}"}})
            return
        running = self._board.begin(identity.purpose, identity.ref_id, identity.traceparent,
                                    prepared.preflight.prompt_tokens)
        outcome, stats = "failed", {}
        try:
            if warmup:
                result = run_warmup(self._engine, body, identity, prepared,
                                    should_yield=self._admission.foreground_active,
                                    on_progress=lambda event: running.on_prefill(event["prefill"]))
                outcome = "warmed" if result["outcome"] == "done" else result["outcome"]
                stats = {"prompt_tokens": result["prompt_tokens"], "cached_tokens": result["cached_tokens"]}
                _send_json(h, 200, result)
            else:
                result = self._chat_tracked(h, body, identity, running, prepared)
                usage = result["usage"]
                outcome = "cancelled" if result["cancelled"] else result["finish_reason"].replace("tool_calls", "tool call")
                stats = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                         "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"]}
        except ClientGone:
            outcome = "cancelled"
        except Exception as e:  # noqa: BLE001 - the engine fell over; say so instead of dropping the connection
            log.error("LmkRequestFailed", "request failed inside lmk", purpose=identity.purpose,
                      refId=identity.ref_id, error=repr(e))
            if not getattr(h, "lmk_response_started", False):
                _send_json(h, 500, {"error": {"type": "internal_error", "message": f"lmk failed on this request: {e}"}})
            h.close_connection = True
        finally:
            # the one way out: whatever happened above, the place is given back and the request leaves the board
            self._admission.leave(ticket)
            self._board.finish(running, outcome, **stats)

    def _chat_tracked(self, h, body: dict, identity: CallerIdentity, running: Running, prepared) -> dict:
        def on_progress(event: dict) -> None:
            if "prefill" in event:
                running.on_prefill(event["prefill"])
            else:
                running.on_decode(event["decode"]["part"], event["decode"]["completion_tokens"])

        if body.get("stream"):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Connection", "close")
            h.end_headers()
            h.lmk_response_started = True

            def emit(chunk: dict) -> None:
                try:
                    h.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
                    h.wfile.flush()
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise ClientGone() from e

            result = run_chat(self._engine, body, identity, emit, on_progress, prepared)
            if not result["cancelled"]:
                try:
                    h.wfile.write(b"data: [DONE]\n\n")
                    h.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            h.close_connection = True
            return result
        result = run_chat(self._engine, body, identity, lambda _chunk: None, on_progress, prepared)
        message = {"role": "assistant", "content": result["content"] or None}
        if result["reasoning_content"]:
            message["reasoning_content"] = result["reasoning_content"]
        if result["tool_calls"]:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in result["tool_calls"]]
        _send_json(h, 200, {**result["base"], "object": "chat.completion", "usage": result["usage"],
                            "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}]})
        return result

    @property
    def engine(self) -> Engine:
        return self._engine

    def status(self) -> dict:
        m = self._engine.loaded_model()
        board = self._board.snapshot()
        return {
            "build": self._build,
            "engine": engine_commit(),
            "config_fingerprint": self._config_fingerprint,
            "model": {"id": m.id, "path": str(m.path), "context_length": m.context_length,
                      "requested_context_length": m.requested_context_length,
                      "input_modalities": self._engine.input_modalities(),
                      "thinking": self._engine.thinking_enabled(),
                      "reasoning_effort": self._engine.reasoning_effort(),
                      "kv_cache_bits": m.kv_cache_bits,
                      "kv_cache_bits_auto": self._kv_cache_bits_auto,
                      "mac_memory_gb": mac_memory_gb(get_current_memory().read().total_bytes),
                      "speculative_decoding": getattr(m, "speculative_decoding", False),
                      "draft_kind": getattr(m, "draft_kind", None)},
            "sampling_defaults": self._engine.sampling_defaults(),
            "draft": getattr(self._engine, "draft_stats", lambda: None)(),
            "cache": self._engine.cache_stats(),
            "memory": self._memory(),
            "requests": self._admission.counts(),
            "totals": board["totals"],
            "in_flight": board["in_flight"],
            "waiting": self._admission.waiting(),
            "recent": board["recent"],
            "uptime_ms": get_current_clock().mono_ms() - self._started_ms,
        }

    def _memory(self) -> dict:
        reading = get_current_memory().read()
        return {"pressure": reading.pressure, "free_percent": reading.free_percent,
                "total_bytes": reading.total_bytes, "lmk_gpu_bytes": self._engine.gpu_memory_bytes(),
                # MLX's peak counts memory in use only, while lmk_gpu_bytes also counts its buffers: the two are
                # not comparable (seen live: "holds 23.3 GB (peak 20.9 GB)"), so the name says which it is
                "lmk_gpu_peak_in_use_bytes": self._engine.gpu_memory_peak_bytes()}


def _send_json(h: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    raw = json.dumps(body, ensure_ascii=False).encode()
    h.lmk_response_started = True
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(raw)))
    h.end_headers()
    h.wfile.write(raw)
