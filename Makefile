# lmk: developer targets. People who just want to use lmk run install.sh (see README.md).

PYTHON311 ?= $(HOME)/.local/bin/python3.11
VENV      := .venv
PY        := $(VENV)/bin/python
ENGINE    := .engine/mlx-engine
ENGINE_COMMIT := $(shell cat ENGINE_COMMIT)
export PYTHONPATH := $(CURDIR)/$(ENGINE):$(CURDIR)

.DEFAULT_GOAL := help

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

all: lint test ## the pre-commit sweep

venv: $(VENV)/.installed $(ENGINE)/.pinned ## python env + the engine at its pinned commit

$(VENV)/.installed: requirements.txt
	$(PYTHON311) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	touch $@

# mlx-engine ships no packaging metadata, so it is cloned, not pip-installed.
$(ENGINE)/.pinned: ENGINE_COMMIT
	mkdir -p .engine
	[ -d $(ENGINE) ] || git clone --quiet https://github.com/seabit-ai/mlx-engine.git $(ENGINE)
	git -C $(ENGINE) fetch --quiet origin
	git -C $(ENGINE) checkout --quiet $(ENGINE_COMMIT)
	touch $@

test: venv ## unit tests (no GPU, no model)
	$(PY) -m pytest -q tests

itest: venv ## integration tests — need the configured model on disk and a free GPU
	LMK_ITEST=1 $(PY) -m pytest -q tests -m itest

cache-fixture: venv ## before an engine upgrade: write a cache with the CURRENT engine
	rm -rf .compat-cache
	LMK_ITEST=1 LMK_COMPAT_DIR=$(CURDIR)/.compat-cache $(PY) -m pytest -q tests/test_itest_cache_compat.py

cache-compat: venv ## after an engine upgrade: the new engine must restore that cache
	@[ -f .compat-cache/expected.json ] || { echo "no fixture — run 'make cache-fixture' with the old engine first"; exit 1; }
	LMK_ITEST=1 LMK_COMPAT_DIR=$(CURDIR)/.compat-cache $(PY) -m pytest -q tests/test_itest_cache_compat.py

lint: venv ## byte-compile everything
	$(PY) -m compileall -q lmk tests

run: venv ## run the server in the foreground from this tree, with ~/.lmk/config.yaml
	$(PY) -m lmk serve

clean: ## remove the env and the engine clone
	rm -rf $(VENV) .engine .pytest_cache

# ---- the resident service ----
# The same installer users run, fed from this working tree. The service runs from
# ~/.lmk/app, never from here: switching branches must not take the server down.
install: ## install this tree into ~/.lmk and (re)start the service
	./install.sh
	$${LMK_HOME:-$(HOME)/.lmk}/bin/lmk up

uninstall: ## stop the service (keeps ~/.lmk: app, config, cache)
	$${LMK_HOME:-$(HOME)/.lmk}/bin/lmk down

.PHONY: help all venv test itest cache-fixture cache-compat lint run clean install uninstall
