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

"""Label-free identity of the actual arrays, endpoints and fixed graph used by direct execution."""

from passage_eval.execution.runner import request_identity
from passage_eval.io import pinned

FIELDS = ("graph_id", "cost_model", "cost_weight", "start", "goal", "shape", "elevation_hash", "obstacles_hash")


def raw_identity_from_full(identity):
    if any(field not in identity for field in FIELDS):
        raise ValueError("raw identity fields missing")
    return {field: identity[field] for field in FIELDS}


def raw_identity(request):
    return raw_identity_from_full(request_identity(request))


def raw_digest(request):
    return pinned.canonical_sha256(raw_identity(request))


def task_from_full(sample_id, identity):
    return {"sample_id": sample_id, "raw_input_sha256": pinned.canonical_sha256(raw_identity_from_full(identity))}
