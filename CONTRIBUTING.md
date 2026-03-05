# Contributing guidelines

Thank you for your interest in contributing to PASSAGE.

This document explains team organization, communication channels, and the contribution rules to follow before opening a pull request.

## Team organization

`passage` repository is maintained by the Thales open-source team with a designated project lead.

Project communication channels:
- GitHub Issues for bugs and concrete tasks.
- GitHub Discussions for ideas and questions.
- Code of conduct reports via `geoffrey.delhomme@thalesgroup.com` or `oss@thalesgroup.com`.

### Roles

- Project lead: defines technical direction, release priorities, and final arbitration.
- Maintainers: triage issues, review pull requests, and merge accepted changes.
- Contributors: propose and implement improvements (code, docs, tests, tooling).

## How to become a contributor

1. Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
2. Fork and clone the repository.
3. Set up the development environment and hooks:

```bash
uv sync --group dev
uv run pre-commit install
```

4. Create a focused `feature/...` branch from `develop`.
5. Implement your change with tests and documentation updates when relevant.
6. Run checks locally, then open a pull request.

### Contributor License Agreements

There is currently no separate CLA file in this repository.

By contributing, you confirm that:
- You have the right to submit the contribution.
- Your contribution can be distributed under this repository's MIT License.

### Contributing code

Use small, focused commits and keep one logical change per pull request.

Recommended local workflow:

```bash
git checkout develop
git pull --ff-only
git checkout -b feature/your-change
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest --cov
uv run pre-commit run --all-files
```

## Gitflow workflow

`passage` uses the Gitflow branching model.

![Gitflow workflow diagram](https://dam-cdn.atl.orangelogic.com/AssetLink/t8b1bnptx6bn40wc43g83j02u5b61064.svg)

Reference: [Atlassian Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

Branch roles:
- `main`: production-ready history and release tags.
- `develop`: integration branch for the next release.
- `feature/<name>`: short-lived work branch created from `develop`, merged back into `develop`.
- `release/<version>`: release preparation branch created from `develop`, then merged into both `main` and `develop`.
- `hotfix/<name>`: urgent fix branch created from `main`, then merged into both `main` and `develop`.

Contributor rules with Gitflow:
- Standard contributions must be opened as pull requests to `develop`.
- Use `feature/<name>` for regular contributions.
- Use `hotfix/<name>` only for urgent production fixes validated by maintainers.
- Keep your branch synchronized with `develop` while your pull request is open.

## Pull Request Checklist

Before requesting review, verify all items below.

### License

- [ ] My contribution is compatible with the MIT License in this repository.
- [ ] I did not add third-party code or assets without proper licensing/attribution.

### Coding style

- [ ] `uv run ruff check src/ tests/` passes.
- [ ] `uv run ruff format src/ tests/` has been applied.
- [ ] `uv run pre-commit run --all-files` passes.
- [ ] New code follows existing naming, structure, and readability conventions.
- [ ] PR targets `develop` (or `main` only for approved hotfixes).

### Testing

- [ ] Relevant tests were added or updated.
- [ ] Existing tests pass locally.
- [ ] Coverage remains at or above the project threshold (80%).

#### Running sanity check

```bash
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
```

#### Running unit tests

```bash
uv run pytest -m unit
uv run pytest --cov
```

### Issues management

- Search existing issues/discussions before opening a new thread.
- Open one issue per bug or feature request to keep tracking clear.
- For bug reports, include environment, steps to reproduce, expected behavior, and actual behavior.
- Keep discussions technical and respectful, following the Code of Conduct.
