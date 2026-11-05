# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
