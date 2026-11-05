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

"""Native warm-up disjointness uses synthetic input identities only."""

import pytest
from passage_eval.io import roster as roster_io
from passage_eval.protocol import admission as A


def task(sample_id, seed):
    identity = {key: f"{seed:064x}" for key in roster_io.CONTENT_IDENTITY_FIELDS}
    return {"sample_id": sample_id, "input_identity": identity}


def test_fixed_warmup_can_be_reused_with_another_evaluated_roster():
    warmup = {"tasks": [task("warm", 1)]}
    measured = {"tasks": [task("a", 2), task("b", 3)]}
    assert A.native_warmup_disjoint(warmup, measured)


@pytest.mark.parametrize("collision", ["id", "content"])
def test_fixed_warmup_rejects_id_or_input_content_collision(collision):
    warmup = {"tasks": [task("warm", 1)]}
    other = task("warm" if collision == "id" else "other", 1 if collision == "content" else 2)
    assert not A.native_warmup_disjoint(warmup, {"tasks": [other]})
