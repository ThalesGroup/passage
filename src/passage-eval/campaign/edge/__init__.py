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

"""The minimal input-only boundary for native edge qualification.

This package intentionally has no lease, service, telemetry, or host-inference
implementation.  Targets receive only sealed ordinary inputs; references and
scores remain with the host collector.
"""

from __future__ import annotations

PACKAGE_SCHEMA = "passage-edge-input-package/v1"
TARGET_RESULT_SCHEMA = "passage-edge-target-result/v1"
NATIVE_ADAPTER_SCHEMA = "passage-edge-native-adapter/v1"
FINAL_DECODERS_SHA256 = "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
INPUT_MEMBERS = (
    "sample_ids",
    "terrain",
    "obstacles",
    "starts",
    "goals",
    "input_tensor",
)


class EdgeRelayError(ValueError):
    """A target input, receipt, or collected output is not safe to score."""


class PackageError(EdgeRelayError):
    """A sealed input-only package is malformed, changed, or discloses private data."""


__all__ = [
    "EDGE_TARGETS",
    "FINAL_DECODERS_SHA256",
    "INPUT_MEMBERS",
    "NATIVE_ADAPTER_SCHEMA",
    "PACKAGE_SCHEMA",
    "TARGET_RESULT_SCHEMA",
    "EdgeRelayError",
    "PackageError",
]

EDGE_TARGETS = ("thor", "orin", "tiger")
