# lmk (lm-kitten): local LLM server on the open-source mlx-engine.
# Design: ../docs/design/2026-09-19-lmk.md

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
	[ -d $(ENGINE) ] || git clone --quiet https://github.com/lmstudio-ai/mlx-engine.git $(ENGINE)
	git -C $(ENGINE) fetch --quiet origin
	git -C $(ENGINE) checkout --quiet $(ENGINE_COMMIT)
	touch $@

test: venv ## unit tests (no GPU, no model)
	$(PY) -m pytest -q tests

itest: venv ## integration tests — need the configured model on disk and a free GPU
	LMK_ITEST=1 $(PY) -m pytest -q tests -m itest

lint: venv ## byte-compile everything
	$(PY) -m compileall -q lmk tests

run: venv ## start the server with ~/.kitten/lmk.yaml
	$(PY) -m lmk

clean: ## remove the env and the engine clone
	rm -rf $(VENV) .engine .pytest_cache

# ---- the resident service ----
# Runs from PREFIX, never from this working tree: switching branches in the repo
# must not take the server down.
PREFIX ?= $(HOME)/.kitten/lmk/app
LABEL  := ai.kitten.lmk
PLIST  := $(HOME)/Library/LaunchAgents/$(LABEL).plist
LOGDIR := $(HOME)/Library/Logs/kitten

install: ## install to PREFIX and (re)start the launchd service
	@[ -f $(HOME)/.kitten/lmk.yaml ] || { echo "no ~/.kitten/lmk.yaml — see README.md"; exit 1; }
	mkdir -p $(PREFIX) $(LOGDIR)
	rsync -a --delete --exclude __pycache__ lmk $(PREFIX)/
	cp Makefile requirements.txt ENGINE_COMMIT $(PREFIX)/
	$(MAKE) -C $(PREFIX) venv
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LOGDIR@|$(LOGDIR)|g' launchd.plist.in > $(PLIST)
	-launchctl bootout gui/$$(id -u)/$(LABEL) 2>/dev/null
	launchctl bootstrap gui/$$(id -u) $(PLIST)
	@echo "installed; status: curl -s http://127.0.0.1:<port>/lmk/v1/status   logs: $(LOGDIR)/lmk.jsonl"

uninstall: ## stop the service and remove the LaunchAgent (keeps PREFIX, config and cache)
	-launchctl bootout gui/$$(id -u)/$(LABEL)
	rm -f $(PLIST)

.PHONY: help all venv test itest lint run clean install uninstall
