import os
import sys
from pathlib import Path

from lmk import log
from lmk.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from lmk.engine import MlxEngine
from lmk.server import LmkServer


def main() -> int:
    try:
        cfg = load_config(Path(os.environ.get("LMK_CONFIG") or DEFAULT_CONFIG_PATH))
    except ConfigError as e:
        print(f"lmk: {e}", file=sys.stderr)
        return 2
    log.open_log_file(cfg.log_dir)
    log.info("LmkStarting", "loading the resident model", model=cfg.model.id,
             path=str(cfg.model.path), contextLength=cfg.model.context_length)
    engine = MlxEngine(cfg.model.id, cfg.model.path, cfg.model.context_length)
    server = LmkServer(engine, cfg.host, cfg.port)
    log.info("LmkReady", "serving", host=cfg.host, port=server.port, model=cfg.model.id)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("LmkStopping", "interrupted")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
