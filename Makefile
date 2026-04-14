SHELL := /bin/bash

PASSAGE ?= uv run passage
CONFIG ?= config/config.yaml
TILES_DIR ?= outputs/download/tiles
CALIBRATE_DIR ?= outputs/calibrate
DATASET_DIR ?= outputs/dataset
EXPORT_DIR ?= outputs/export
CALIBRATE_BACKUP ?= $(CALIBRATE_DIR)/calibrate.json.bak
RESOLUTION ?=
NUM_CPUS ?=
REPO_ID ?=

.PHONY: install sync download calibrate restore-calibrate generate export export-local manifest rebuild test lint

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

export-local:
	$(PASSAGE) export --config $(CONFIG) --dataset-dirpath $(DATASET_DIR) --calibrate-dirpath $(CALIBRATE_DIR) --no-upload $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),)

export:
	$(PASSAGE) export --config $(CONFIG) --dataset-dirpath $(DATASET_DIR) --calibrate-dirpath $(CALIBRATE_DIR) $(if $(strip $(RESOLUTION)),--resolution $(RESOLUTION),) $(if $(strip $(NUM_CPUS)),--num-cpus $(NUM_CPUS),) $(if $(strip $(REPO_ID)),--repo-id $(REPO_ID),)

manifest:
	mkdir -p $(EXPORT_DIR)
	find $(EXPORT_DIR) \( -path "$(EXPORT_DIR)/.cache" -o -path "$(EXPORT_DIR)/.cache/*" \) -prune -o -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum > $(EXPORT_DIR)/manifest.sha256

rebuild: download calibrate generate export-local manifest

test:
	@status=0; \
	uv run pytest --cov || status=$$?; \
	$(MAKE) restore-calibrate; \
	exit $$status

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
