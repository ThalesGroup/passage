---
title: Testing Strategy
---

# Testing Strategy

PASSAGE maintains a comprehensive test suite with **23 test files**, targeting **80%+ code coverage**. Tests cover unit logic, integration workflows, CLI behavior, and edge cases.

---

## Test Organization

```
tests/
├── conftest.py                          # Shared fixtures
├── test_calibrate.py                    # Calibrate unit tests
├── test_calibrate_integration.py        # Calibrate end-to-end
├── test_calibrate_no_overwrite.py       # Overwrite protection
├── test_cli.py                          # CLI argument parsing
├── test_config.py                       # Config loading & validation
├── test_download.py                     # Download orchestration
├── test_download_extract.py             # ZIP extraction logic
├── test_download_tile.py                # Single tile download
├── test_execute_notebook_uv_run.py      # Notebook execution
├── test_export_retries.py               # Export retry logic
├── test_generate.py                     # Generation pipeline
├── test_generate_costmodels.py          # Cost model correctness
├── test_generate_helpers.py             # Generation utilities
├── test_large_resolution.py             # Multi-tile assembly
├── test_no_images.py                    # --no-images flag
├── test_num_samples_per_resolution.py   # Per-resolution sample counts
├── test_obstacle_generation.py          # Obstacle mask creation
├── test_obstacles_distribution.py       # Shape distribution stats
├── test_obstacles.py                    # Superellipse correctness
├── test_prepare_hf.py                   # HF export preparation
├── test_prepare_images_parquet.py       # Image in Parquet tests
└── test_prepare_notebook_retry.py       # Notebook retry edge cases
```

---

## Pytest Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "download: Tests requiring network access",
    "calibrate: Calibration tests",
    "generate: Generation tests"
]

[tool.coverage.run]
source = ["src/passage"]
branch = true

[tool.coverage.report]
fail_under = 80
```

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
    # Unit tests only
    uv run pytest -m unit

    # Skip network-dependent tests
    uv run pytest -m "not download"
    ```

=== "Parallel execution"
    ```bash
    # Run with 4 workers (requires pytest-xdist)
    uv run pytest -n 4
    ```

---

## Test Fixtures

The `conftest.py` provides shared fixtures:

| Fixture | Description |
|---------|-------------|
| Mock tile statistics | Simulated `calibrate.json` data |
| Mock calibration data | Pre-computed global min/max |
| Sample configuration | Minimal `config.yaml` for testing |
| Temporary directories | Isolated output paths for test safety |

!!! note "Test safety"
    When `PYTEST_CURRENT_TEST` is set (automatically by pytest), output-producing modules redirect to temporary directories rather than the real `outputs/` folder.

---

## Coverage Targets

| Module | Coverage target | Notes |
|--------|:--------------:|-------|
| `utils.py` | 95%+ | Small, well-defined functions |
| `obstacles.py` | 95%+ | Pure math, easy to test |
| `calibrate.py` | 90%+ | Mocked I/O |
| `download.py` | 80%+ | Network mocking needed |
| `generate.py` | 80%+ | Complex, many code paths |
| `pathfinding_utils.py` | 80%+ | Multiple backends |
| `export.py` | 75%+ | Heavy I/O, HF API mocking |
| `visualization.py` | 70%+ | Visual output, harder to assert |

The project-wide **minimum threshold is 80%** branch coverage. CI fails if coverage drops below this level.

---

## Quality Tools

### Ruff

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting:

```bash
# Lint check
uv run ruff check src/

# Auto-fix
uv run ruff check src/ --fix

# Format
uv run ruff format src/
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
indent-width = 4
```

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- `uv.lock` sync check
- Ruff lint + format

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

---

## CI Integration

Tests run automatically on every push and pull request via GitHub Actions. See the [CI/CD workflow](https://github.com/ThalesGroup/passage/blob/main/.github/workflows/ci.yml) for details.

The CI pipeline:

1. **Lint** — `ruff check` + `ruff format --check`
2. **Test** — `pytest` with coverage reporting
3. **Coverage gate** — fail if below 80%
