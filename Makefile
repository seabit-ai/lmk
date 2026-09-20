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

.PHONY: help all venv test itest lint run clean
