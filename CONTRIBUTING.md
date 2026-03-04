# Contributing to PASSAGE

Thank you for considering contributing to PASSAGE! Whether it's reporting a bug, proposing a feature, improving documentation, or writing code — every contribution helps advance pathfinding research.

For the **full contributing guide** (code style, testing, PR workflow, branch conventions), please see the [online documentation](https://thalesgroup.github.io/passage/contributing/).

---

## Quick Start

```bash
# 1. Fork & clone
git clone https://github.com/<your-username>/passage.git
cd passage

# 2. Install dev dependencies
uv sync --group dev

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Create a branch
git checkout -b feature/your-feature-name
```

---

## Development Workflow

```bash
# Run tests (80 %+ coverage required)
uv run pytest --cov

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run pre-commit on all files
uv run pre-commit run --all-files
```

---

## Reporting Bugs & Requesting Features

- **Bug reports**: [open an issue](https://github.com/ThalesGroup/passage/issues/new?template=bug_report.md)
- **Feature requests**: [start a discussion](https://github.com/ThalesGroup/passage/discussions/categories/ideas)

Please search existing issues and discussions before creating a new one.

---

## Code of Conduct

By participating in this project you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
