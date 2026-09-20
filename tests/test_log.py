import json

from lmk import log
from lmk.clock import get_current_clock, set_current_clock


class FixedClock:
    def wall_ms(self): return 1789840000000
    def mono_ms(self): return 5


def test_every_line_is_json_with_an_event_and_lands_in_the_file(tmp_path, capsys):
    previous = get_current_clock()
    set_current_clock(FixedClock())
    try:
        log.open_log_file(tmp_path)
        log.info("LmkReady", "serving", port=1235)
    finally:
        set_current_clock(previous)
    line = (tmp_path / "lmk.jsonl").read_text().strip()
    assert json.loads(line) == {"time_ms": 1789840000000, "level": "INFO", "event": "LmkReady", "msg": "serving", "port": 1235}
    assert json.loads(capsys.readouterr().err.strip()) == json.loads(line)
