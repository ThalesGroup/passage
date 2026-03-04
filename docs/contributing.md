---
title: Contributing
icon: material/hand-heart
---

# Contributing to PASSAGE

Thank you for considering contributing to PASSAGE! Whether it's reporting a bug, proposing a feature, improving documentation, or writing code — every contribution helps advance pathfinding research.

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/passage.git
cd passage
```

### 2. Set Up Development Environment

```bash
# Install all dependencies including dev tools
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

!!! tip "Branch naming"
    Use descriptive prefixes: `feature/`, `fix/`, `docs/`, `test/`, `refactor/`.

---

## Development Workflow

### Code Style

PASSAGE uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

Style rules:

- **Line length:** 120 characters
- **Indent:** 4 spaces
- **Docstrings:** Google style
- **Type hints:** encouraged but not enforced

### Running Tests

```bash
# All tests
uv run pytest

# With coverage report
uv run pytest --cov --cov-report=html

# Specific test file
uv run pytest tests/test_obstacles.py -v

# Skip slow / network tests
uv run pytest -m "not download"
```

Tests must pass and coverage must remain **≥ 80%** before merging.

### Pre-commit Hooks

Hooks run automatically on `git commit`:

- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- `uv.lock` sync check
- Ruff lint + format

To run manually:

```bash
uv run pre-commit run --all-files
```

---

## Pull Request Guidelines

1. **Keep PRs focused** — one feature or fix per PR
2. **Write tests** for new functionality
3. **Update documentation** if behavior changes
4. **Add type hints** to new functions
5. **Follow existing patterns** in the codebase
6. **Write descriptive commit messages**

### PR Template

```markdown
## What does this PR do?
Brief description of changes.

## How was it tested?
- [ ] New unit tests added
- [ ] Existing tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style (ruff check/format)
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Pre-commit hooks pass
```

---

## Reporting Bugs

When reporting a bug, please include:

1. **Python version** and **OS**
2. **PASSAGE version** (`passage --help` shows the CLI)
3. **Steps to reproduce** the issue
4. **Expected vs. actual behavior**
5. **Full error traceback** (if applicable)
6. **Configuration** (relevant sections of `config/config.yaml`)

[Open a bug report →](https://github.com/ThalesGroup/passage/issues/new?labels=bug)

---

## Proposing Features

We welcome feature proposals! To propose a new feature:

1. Check [existing issues](https://github.com/ThalesGroup/passage/issues) to avoid duplicates
2. Open a new issue with the `enhancement` label
3. Describe the **use case** and **expected behavior**
4. If possible, outline a **proposed implementation**

[Propose a feature →](https://github.com/ThalesGroup/passage/issues/new?labels=enhancement)

---

## Areas for Contribution

Here are some areas where contributions are especially welcome:

| Area | Examples |
|------|---------|
| **New cost models** | Time-based, wind-aware, fuel consumption |
| **New solvers** | RRT\*, D\*, Theta\* |
| **Benchmarks** | ML baselines (U-Net, GNN, PPO) |
| **Visualization** | Interactive 3D terrain viewer |
| **Documentation** | Tutorials, examples, translations |
| **Performance** | GPU acceleration, improved Numba kernels |
| **Data sources** | Additional elevation datasets, land cover |

---

## Code of Conduct

Be respectful, inclusive, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## License

By contributing to PASSAGE, you agree that your contributions will be licensed under the [MIT License](https://github.com/ThalesGroup/passage/blob/main/LICENSE).
