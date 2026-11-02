---
title: Testing Strategy
---

# Testing Strategy

PASSAGE has **65 test files**: 44 in `tests/` for the `passage` package and 21 in `tests/passage_eval/` for the evaluator. Together they run **865 tests**. Coverage is measured on both packages, with a gate at **80 %**. The tests cover unit logic, integration workflows, CLI behavior and edge cases.

---

## Test Organization

Shared fixtures live in `tests/conftest.py` and `tests/passage_eval/conftest.py`.

| Area | Test files in `tests/` |
|------|------------------------|
| CLI | `test_cli.py`, `test_analysis_cli.py` |
| Configuration | `test_config.py`, `test_num_samples_per_resolution.py` |
| Download | `test_download.py`, `test_download_extract.py`, `test_download_integrity.py`, `test_download_tile.py` |
| Calibration | `test_calibrate.py`, `test_calibrate_integration.py`, `test_calibrate_no_overwrite.py`, `test_calibrate_provenance.py` |
| Generation | `test_generate.py`, `test_generate_costmodels.py`, `test_generate_helpers.py`, `test_large_resolution.py`, `test_no_images.py`, and one file for the release generator |
| Obstacles | `test_obstacles.py`, `test_obstacle_generation.py`, `test_obstacles_distribution.py` |
| Solvers | `test_pathfinding_utils.py`, `test_reference.py` |
| Export and publication | `test_export_keep_configs.py`, `test_export_retries.py`, `test_prepare_hf.py`, `test_prepare_images_parquet.py`, `test_prepare_notebook_retry.py`, and two files for the release export and its publication metadata |
| Full rebuild | `test_regenerate.py` |
| Dataset access, audit and qualification | `test_dataset_source.py`, `test_audit.py`, `test_authentication.py`, `test_qualification.py`, `test_qualification_runner.py`, `test_qualification_reports.py` |
| Notebooks | `test_execute_notebook_uv_run.py`, `test_notebook_export.py`, `test_notebook_reports.py`, `test_notebook_templates.py` |
| Plots | `test_plotting.py`, `test_visualization.py`, `test_trajectory_style.py` |

The evaluator tests in `tests/passage_eval/` cover its contracts, metrics (`test_regret.py`, `test_geometric.py`, `test_reliability.py`), statistics, execution and timing, results format, CLI and the paper contract (`test_paper_contract.py`).

---

## Pytest Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--html=outputs/tests/report.html", "--self-contained-html", "--import-mode=importlib"]
markers = [
  "unit: Unit tests (fast, isolated)",
  "integration: Integration tests (may require external resources)",
  "download: Tests for download module",
  "calibrate: Tests for calibrate module",
  "generate: Tests for generate module"
]

[tool.coverage.run]
source = ["src/passage", "src/passage-eval"]
branch = true
omit = ["*/tests/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

Every run also writes an HTML report to `outputs/tests/report.html`.

---

## Running Tests

=== "All tests"
    ```bash
    uv run pytest
    ```

=== "With coverage"
    ```bash
    uv run pytest --cov --cov-report=html
    ```

=== "Specific markers"
    ```bash
    # Tests marked as unit tests only
    uv run pytest -m unit

    # Skip the tests of the download module
    uv run pytest -m "not download"
    ```

    Markers are set on part of the suite only, so `-m unit` runs a subset.

=== "Parallel execution"
    ```bash
    # Run with 4 workers (pytest-xdist is in the dev dependency group)
    uv run pytest -n 4
    ```

`make test` runs `uv run pytest --cov`, then restores `outputs/calibrate/calibrate.json` from its backup, if one exists.

---

## Test Fixtures

`tests/conftest.py` provides shared fixtures:

| Fixture | Description |
|---------|-------------|
| `mock_tile_stats_df`, `mock_calibration_data` | Simulated tile statistics and calibration data |
| `calibrate_json_file`, `config_yaml_file` | A `calibrate.json` and a minimal `config.yaml` written under `tmp_path` |
| `mock_tiles_dir` | An empty tiles directory |
| `valid_sample_files`, `partial_sample_files` | Complete and incomplete generated scenes, for resume tests |

`tests/passage_eval/conftest.py` provides `release_row`, a small row in the released format, and two matching `Scene` objects.

!!! note "Test outputs"
    The code writes to the paths it is given. Tests must pass temporary paths, normally through pytest's `tmp_path` fixture; nothing redirects outputs automatically.

---

## Coverage Gate

Coverage is measured with branches on `src/passage` and `src/passage-eval`. The project-wide **minimum is 80 %** (`fail_under = 80`): a run with `--cov` fails below this level.

---

## Quality Tools

### Ruff

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting:

```bash
# Lint check
uv run ruff check src/ tests/

# Auto-fix
uv run ruff check src/ tests/ --fix

# Format
uv run ruff format src/ tests/
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
indent-width = 4
show-fixes = true
```

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

- Trailing whitespace removal (Markdown excluded)
- End-of-file fixer
- YAML validation
- `uv.lock` update when `pyproject.toml` changes
- Ruff lint (with `--fix`) and format
- TOML key sorting

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

---

## CI Integration

Tests run on GitHub Actions for every push and pull request to `main`. See the [CI workflow](https://github.com/ThalesGroup/passage/blob/main/.github/workflows/ci.yml) for details.

The CI pipeline:

1. **Lint** — `ruff check` and `ruff format --check` on `src/` and `tests/`
2. **Test** — `pytest` with coverage, on Python 3.11 and 3.12
3. **Coverage gate** — the run fails if coverage drops below 80 %
