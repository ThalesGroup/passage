# Security Policy

This document describes how security issues are reported, handled, and communicated for PASSAGE.

## Supported Versions

Security fixes are provided on a best-effort basis for older versions.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately.

Preferred contacts:
- `geoffrey.delhomme@thalesgroup.com`
- `oss@thalesgroup.com`

Please include:
- A clear description of the issue and potential impact.
- Affected versions/commit hash.
- Reproduction steps or proof of concept (minimal and safe).
- Any suggested remediation.

Important rules:
- Do not publish exploit details in public issues, pull requests, or discussions.
- If you already opened a public issue by mistake, contact maintainers immediately so details can be redacted.

## Disclosure Policy

Our coordinated disclosure process is:

1. Acknowledge report receipt (target: within 5 business days).
2. Triage severity, impact, and affected scope.
3. Prepare and validate a fix.
4. Coordinate disclosure timing with the reporter when possible.
5. Publish a fix and a public security note once users can mitigate.

For actively exploited or critical issues, timelines may be accelerated.

## Security Update Policy

- Security fixes are released as soon as practical.
- Release notes will clearly identify security-relevant changes.
- Users should upgrade to the latest supported version immediately after a security release.
- Critical fixes may be shipped as dedicated hotfix releases.

## Security-Related Configuration

To reduce security risk when contributing or deploying:

- Use supported Python versions (`>=3.11`) and pinned dependencies (`uv.lock`).
- Keep dependencies updated (`uv sync --group dev`).
- Never commit credentials, API keys, or tokens.
- Keep secrets in environment variables or local secret files (for example `.env.secret`, which is git-ignored).
- Prefer least-privilege credentials (for example short-lived or scoped tokens).
- Run project checks before pushing:

```bash
uv run pre-commit run --all-files
uv run ruff check src/ tests/
uv run pytest --cov
```

If your workflow uses PyTorch or other external ML tooling, follow upstream secure usage guidance and treat all external data/artifacts as untrusted inputs.

## Security Best Practices for Contributors

- Validate external files and datasets before processing.
- Avoid executing untrusted notebooks or scripts.
- Review dependency changes carefully in pull requests.
- Minimize exposed data in logs, issues, and CI output.

## Known Security Gaps and Future Enhancements

Current limitations (contributions welcome):

- No dedicated automated secret-scanning hook is enforced in CI.
- No dedicated automated dependency vulnerability gate is enforced in CI.
- No signed release artifacts/attestations are published yet.
