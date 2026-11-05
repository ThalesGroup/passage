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

"""Bounds on everything a candidate can write into a run artifact.

Diagnostics, error messages, stage names, counter keys and the stderr tail are chosen by the
candidate. They are capped here so that a candidate cannot use them to smuggle bulk data out of
the sandbox through the public run directory, and so that a hostile reply cannot make the
manifest unbounded. Text that contains a protected location (the dataset root) is rewritten.
Run artifacts still carry candidate-controlled text and must be treated as private until reviewed.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

MAX_DIAGNOSTICS_BYTES = 4096
MAX_ERROR_CHARS = 512
MAX_ERROR_TYPE_CHARS = 64
MAX_STDERR_CHARS = 1024
MAX_STRING_CHARS = 1024
MAX_ITEMS = 64
MAX_DEPTH = 6
MAX_KEY_CHARS = 64
MAX_LABEL_CHARS = 64
REDACTED = "<redacted:protected-location>"
TRUNCATED = "...[truncated]"

LIMITS = {
    "diagnostics_max_bytes": MAX_DIAGNOSTICS_BYTES,
    "error_max_chars": MAX_ERROR_CHARS,
    "stderr_tail_max_chars": MAX_STDERR_CHARS,
    "string_max_chars": MAX_STRING_CHARS,
    "container_max_items": MAX_ITEMS,
    "max_depth": MAX_DEPTH,
    "counter_key_max_chars": MAX_KEY_CHARS,
}


def scrub_text(text: str, forbidden: Iterable[str]) -> str:
    for fragment in forbidden:
        if fragment and fragment in text:
            text = text.replace(fragment, REDACTED)
    return text


def cap_text(text: Any, limit: int, forbidden: Iterable[str] = ()) -> str:
    text = scrub_text(str(text), forbidden)
    return text if len(text) <= limit else text[: max(0, limit - len(TRUNCATED))] + TRUNCATED


def _cap(value: Any, forbidden: tuple[str, ...], depth: int) -> Any:
    if isinstance(value, str):
        return cap_text(value, MAX_STRING_CHARS, forbidden)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= MAX_DEPTH:
        return "<depth-limit>"
    if isinstance(value, Mapping):
        items = list(value.items())
        out = {
            cap_text(key, MAX_KEY_CHARS, forbidden): _cap(item, forbidden, depth + 1) for key, item in items[:MAX_ITEMS]
        }
        if len(items) > MAX_ITEMS:
            out["_truncated_items"] = len(items) - MAX_ITEMS
        return out
    if isinstance(value, (list, tuple)):
        out = [_cap(item, forbidden, depth + 1) for item in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            out.append(f"<{len(value) - MAX_ITEMS} more items>")
        return out
    return cap_text(repr(value), MAX_STRING_CHARS, forbidden)


def cap_json(value: Any, forbidden: Iterable[str] = (), *, max_bytes: int = MAX_DIAGNOSTICS_BYTES) -> Any:
    """A bounded, scrubbed copy of ``value``; whole-value replacement when it is still too large."""
    forbidden = tuple(forbidden)
    capped = _cap(value, forbidden, 0)
    size = len(json.dumps(capped, sort_keys=True, default=str).encode("utf-8"))
    if size > max_bytes:
        return {"_truncated": True, "original_bytes_after_item_caps": size, "limit_bytes": max_bytes}
    return capped


def check_table(name: str, table: Any) -> None:
    """Counter tables must be small mappings with short keys (raises ``ValueError``)."""
    if (
        not isinstance(table, dict)
        or len(table) > MAX_ITEMS
        or any(not isinstance(key, str) or len(key) > MAX_KEY_CHARS for key in table)
    ):
        raise ValueError(f"{name} is not a small mapping with short keys")


def check_label(name: str, value: Any, limit: int = MAX_LABEL_CHARS) -> None:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} is not a short string")
