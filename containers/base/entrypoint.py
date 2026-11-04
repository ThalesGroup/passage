"""Version guard followed by an explicit PASSAGE CLI; no shell interpretation."""
import contextlib
import os
import sys

ALLOWED = {"passage-eval", "passage-inference", "gpu-smoke"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ALLOWED:
        raise SystemExit("expected passage-eval, passage-inference, or gpu-smoke")
    profile = os.environ.get("PASSAGE_PROFILE", "")
    if profile not in {"thor", "orin", "tiger"}:
        raise SystemExit("missing or invalid built-in target profile")
    from guard_runtime import main as guard

    # A device test is deliberate; ordinary CLI help must not start inference.
    arguments = ["--profile", profile] + (["--gpu-smoke"] if sys.argv[1] == "gpu-smoke" else [])
    if sys.argv[1] == "gpu-smoke":
        code = guard(arguments)
    else:
        # Preserve machine-readable stdout for the actual evaluator CLI.
        with contextlib.redirect_stdout(sys.stderr):
            code = guard(arguments)
    if code:
        raise SystemExit(code)
    if sys.argv[1] == "gpu-smoke":
        return
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
