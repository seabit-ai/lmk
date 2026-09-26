"""`lmk serve` — the foreground server. launchd runs this; people run `lmk up`."""
import errno
import signal

from lmk import log
from lmk.config import ConfigError, app_dir, build_id, config_path, fingerprint, load_config
from lmk.configfiles import refresh_example, seed_config
from lmk.models import DraftNotDownloaded, ModelNotDownloaded, draft_kind_for, draft_repo_for, resolve_draft, resolve_model

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

    draft_path, draft_kind = None, None
    if cfg.model.speculative_decoding:
        draft_kind = draft_kind_for(cfg.model.source, cfg.model.draft)
        if draft_repo_for(cfg.model.source, cfg.model.draft) is None:
            wanted = f" of kind {cfg.model.draft!r}" if cfg.model.draft else ""
            log.error("LmkConfigInvalid", f"model.speculative_decoding is on, but lmk has no draft model{wanted} for this model "
                      "(the model page says which drafts it has) — set it to false, change model.draft, or pick a model with a draft",
                      path=str(config_path()), model=cfg.model.id)
            return EXIT_WILL_NOT_FIX_ITSELF
        try:
            draft_path = resolve_draft(cfg.model.source, cfg.model.draft)
        except DraftNotDownloaded as e:
            log.error("LmkDraftMissing", f"{e} — run `lmk pull`, then `lmk up`")
            return EXIT_WILL_NOT_FIX_ITSELF
    from lmk.modelfit import why_it_does_not_fit

    too_big = why_it_does_not_fit(resolved.path)
    if too_big:
        log.error("LmkModelDoesNotFit", too_big.replace("\n  ", " "), path=str(resolved.path))
        return EXIT_WILL_NOT_FIX_ITSELF

    from lmk.engine import BUFFER_CACHE_LIMIT_BYTES, MlxEngine, runtime_import_error
    from lmk.server import LmkServer

    not_importable = runtime_import_error()
    if not_importable:
        log.error("LmkRuntimeMissing", f"the model runtime (mlx-engine) is not installed: {not_importable} — "
                  "run the installer again (`make install` from a checkout, or the curl | sh line in the README)",
                  runtimeDir=str(app_dir() / ".engine" / "mlx-engine"))
        return EXIT_WILL_NOT_FIX_ITSELF

    log.info("LmkStarting", "loading the resident model", model=cfg.model.id, path=str(resolved.path),
             requestedContextLength=cfg.model.context_length, build=build_id(),
             kvCacheBits=cfg.model.kv_cache_bits, kvCacheBitsAuto=cfg.model.kv_cache_bits_auto,
             speculativeDecoding=cfg.model.speculative_decoding,
             draftPath=None if draft_path is None else str(draft_path), draftKind=draft_kind, thinking=cfg.model.thinking,
             reasoningEffort=cfg.model.reasoning_effort)
    engine = MlxEngine(cfg.model.id, resolved.path, cfg.model.context_length, cache_dir=cfg.cache_dir,
                       cache_max_bytes=cfg.cache_max_bytes, repo=cfg.model.source.repo, revision=resolved.revision,
                       max_parallel=cfg.requests.max_parallel, template_kwargs=cfg.model.template_kwargs(),
                       kv_cache_bits=cfg.model.kv_cache_bits, draft_path=draft_path, draft_kind=draft_kind,
                       draft_tokens=cfg.model.draft_tokens)
    try:
        # a value the template rejects (Qwen3.8 accepts exactly xhigh / medium / low for reasoning_effort)
        # must stop the start with a clean exit, not the first request with a 500 — and not a restart loop
        engine.chat_format().render([{"role": "user", "content": "probe"}], None)
    except Exception as e:  # noqa: BLE001 - the template raises its own exception type
        log.error("LmkConfigInvalid", f"the model's chat template rejects model.thinking / model.reasoning_effort: {e}",
                  path=str(config_path()), templateKwargs=cfg.model.template_kwargs())
        engine.close()
        return EXIT_WILL_NOT_FIX_ITSELF
    model = engine.loaded_model()
    if model.context_length < (model.requested_context_length or 0):
        log.warn("LmkContextLowered", "not enough memory for the requested context; using a shorter one",
                 requested=model.requested_context_length, inUse=model.context_length)
    try:
        server = LmkServer(engine, cfg.host, cfg.port, build=build_id(),
                           config_fingerprint=fingerprint(cfg, resolved.revision), requests=cfg.requests,
                           kv_cache_bits_auto=cfg.model.kv_cache_bits_auto)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        log.error("LmkPortTaken", f"{cfg.host}:{cfg.port} is already in use — `lmk up` says by whom")
        engine.close()
        return EXIT_WILL_NOT_FIX_ITSELF
    log.info("LmkReady", "serving", host=cfg.host, port=server.port, model=cfg.model.id,
             contextLength=model.context_length, maxParallel=cfg.requests.max_parallel,
             tokenBudget=engine.token_budget(), kvCacheBits=model.kv_cache_bits, kvCacheBitsAuto=cfg.model.kv_cache_bits_auto,
             speculativeDecoding=model.speculative_decoding, draftKind=model.draft_kind, thinking=engine.thinking_enabled(),
             reasoningEffort=engine.reasoning_effort(), bufferCacheLimitBytes=BUFFER_CACHE_LIMIT_BYTES)

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
