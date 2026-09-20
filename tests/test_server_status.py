import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lmk.engine import FakeEngine, LoadedModel
from lmk.server import LmkServer


@pytest.fixture
def server():
    engine = FakeEngine(LoadedModel(id="kitten-27b", path=Path("/m/x"), context_length=200000))
    srv = LmkServer(engine, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def get(srv, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=5) as r:
        return r.status, json.loads(r.read())


def test_status_reports_the_resident_model(server):
    status, body = get(server, "/lmk/v1/status")
    assert status == 200
    assert body["model"] == {"id": "kitten-27b", "path": "/m/x", "context_length": 200000,
                             "input_modalities": ["text"]}
    assert body["in_flight"] == []
    assert body["uptime_ms"] >= 0


def test_openai_models_lists_exactly_the_resident_model(server):
    _, body = get(server, "/v1/models")
    assert [m["id"] for m in body["data"]] == ["kitten-27b"]


def test_unknown_route_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/nope")
    assert e.value.code == 404
    assert json.loads(e.value.read())["error"]["type"] == "not_found"
