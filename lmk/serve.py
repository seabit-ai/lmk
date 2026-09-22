"""`lmk serve` — the foreground server. launchd runs this; people run `lmk up`."""
import errno
import signal

from lmk import log
from lmk.config import ConfigError, build_id, config_path, fingerprint, load_config
from lmk.configfiles import refresh_example, seed_config
from lmk.models import ModelNotDownloaded, resolve_model

# launchd restarts us after a non-zero exit. Problems a restart cannot fix must
# therefore end in a CLEAN exit, or the service spins forever in the background.
EXIT_WILL_NOT_FIX_ITSELF = 0


def serve() -> int:
    seed_config(config_path())
    refresh_example(config_path())
    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("LmkConfigInvalid", str(e), path=str(config_path()))
        return EXIT_WILL_NOT_FIX_ITSELF
    log.open_log_file(cfg.log_dir)
    try:
        resolved = resolve_model(cfg.model.source)
    except (ModelNotDownloaded, FileNotFoundError) as e:
        log.error("LmkModelMissing", f"{e} — run `lmk pull`, then `lmk up`")
        return EXIT_WILL_NOT_FIX_ITSELF

    from lmk.modelfit import why_it_does_not_fit

    too_big = why_it_does_not_fit(resolved.path)
    if too_big:
        log.error("LmkModelDoesNotFit", too_big.replace("\n  ", " "), path=str(resolved.path))
        return EXIT_WILL_NOT_FIX_ITSELF

    from lmk.engine import MlxEngine
    from lmk.server import LmkServer

    log.info("LmkStarting", "loading the resident model", model=cfg.model.id, path=str(resolved.path),
             requestedContextLength=cfg.model.context_length, build=build_id())
    engine = MlxEngine(cfg.model.id, resolved.path, cfg.model.context_length, cache_dir=cfg.cache_dir,
                       cache_max_bytes=cfg.cache_max_bytes, repo=cfg.model.source.repo, revision=resolved.revision,
                       max_parallel=cfg.requests.max_parallel, template_kwargs=cfg.model.template_kwargs())
    model = engine.loaded_model()
    if model.context_length < (model.requested_context_length or 0):
        log.warn("LmkContextLowered", "not enough memory for the requested context; using a shorter one",
                 requested=model.requested_context_length, inUse=model.context_length)
    try:
        server = LmkServer(engine, cfg.host, cfg.port, build=build_id(),
                           config_fingerprint=fingerprint(cfg, resolved.revision), requests=cfg.requests)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        log.error("LmkPortTaken", f"{cfg.host}:{cfg.port} is already in use — `lmk up` says by whom")
        engine.close()
        return EXIT_WILL_NOT_FIX_ITSELF
    log.info("LmkReady", "serving", host=cfg.host, port=server.port, model=cfg.model.id,
             contextLength=model.context_length, maxParallel=cfg.requests.max_parallel,
             tokenBudget=engine.token_budget())

    def stop(signum, _frame):  # launchd stops us with SIGTERM
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("LmkStopping", "shutting down; flushing the prompt cache to disk")
        server.shutdown()
        engine.close()
    return 0
