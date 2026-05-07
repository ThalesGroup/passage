# MIT License
#
# Copyright (c) 2026 THALES

SHELL := /bin/bash

PASSAGE ?= uv run passage
CONFIG ?= config/config.yaml
TILES_DIR ?= outputs/download/tiles
CALIBRATE_DIR ?= outputs/calibrate
DATASET_DIR ?= outputs/dataset
EXPORT_DIR ?= outputs/export
NOTEBOOKS_DIR ?= notebooks
CALIBRATE_BACKUP ?= $(CALIBRATE_DIR)/calibrate.json.bak
RESOLUTION ?=
NUM_CPUS ?=
REPO_ID ?=
DOCS_HOST ?= 127.0.0.1
DOCS_PORT ?= 8000
DOCS_STRICT_DIR ?= /tmp/passage-public

.PHONY: install sync download calibrate restore-calibrate generate compress export upload manifest rebuild test lint docs-build pages-local docs-serve

install:
	bash scripts/install.sh

sync:
	uv sync --group dev

download:
	$(PASSAGE) download --outputs-dirpath outputs/download

calibrate:
	$(PASSAGE) calibrate --tiles-dirpath $(TILES_DIR) --outputs-dirpath $(CALIBRATE_DIR)

restore-calibrate:
	@if [ -f "$(CALIBRATE_BACKUP)" ]; then \
		mkdir -p "$(CALIBRATE_DIR)"; \
		cp "$(CALIBRATE_BACKUP)" "$(CALIBRATE_DIR)/calibrate.json"; \
		echo "Restored $(CALIBRATE_DIR)/calibrate.json from $(CALIBRATE_BACKUP)"; \
	else \
		echo "No calibration backup found at $(CALIBRATE_BACKUP); skipping restore"; \
	fi

generate:
	$(PASSAGE) generate --config $(CONFIG) --tiles-dirpath $(TILES_DIR) --outputs-dirpath $(DATASET_DIR) $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),)

compress:
	$(PASSAGE) compress --config $(CONFIG) --dataset-dirpath $(DATASET_DIR) --calibrate-dirpath $(CALIBRATE_DIR) $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),)

export:
	$(PASSAGE) export --config $(CONFIG) --dataset-dirpath $(DATASET_DIR) --calibrate-dirpath $(CALIBRATE_DIR) --notebooks-dirpath $(NOTEBOOKS_DIR) $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),) $(if $(strip $(REPO_ID)),--repo-id $(REPO_ID),)

upload:
	$(PASSAGE) upload --calibrate-dirpath $(CALIBRATE_DIR) $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),) $(if $(strip $(REPO_ID)),--repo-id $(REPO_ID),)

manifest:
	mkdir -p $(EXPORT_DIR)
	find $(EXPORT_DIR) \( -path "$(EXPORT_DIR)/.cache" -o -path "$(EXPORT_DIR)/.cache/*" \) -prune -o -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum > $(EXPORT_DIR)/manifest.sha256

rebuild: download calibrate generate compress export manifest

test:
	@status=0; \
	uv run pytest --cov || status=$$?; \
	$(MAKE) restore-calibrate; \
	exit $$status

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

docs-build:
	uv run --locked --all-groups mkdocs build --strict --site-dir $(DOCS_STRICT_DIR)

pages-local:
	uv run --locked --all-groups mkdocs build --site-dir public

docs-serve:
	uv run --locked --all-groups mkdocs serve -a $(DOCS_HOST):$(DOCS_PORT)
