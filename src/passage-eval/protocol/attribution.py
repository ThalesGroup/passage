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

"""Declared S2 component controls, distinct from evidence that they were run."""

from collections.abc import Mapping

VARIANTS = frozenset(
    {
        "direct",
        "bounded_repair",
        "prediction_free",
        "exact_fallback",
        "exact_classical",
        "budgeted_classical",
        "oracle_diagnostic",
    }
)


def component_control(value):
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping) or not set(value) <= {
        "variant",
        "guide",
        "checkpoint_id",
        "machinery_id",
        "primary_learned",
    }:
        raise ValueError("component_control requires the finite attribution schema")
    result = dict(value)
    variant, guide = result.get("variant"), result.get("guide")
    if variant not in VARIANTS or guide not in {"learned", "prediction_free", "none", "oracle"}:
        raise ValueError("component_control requires a named variant and guide")
    for key in ("checkpoint_id", "machinery_id"):
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            raise ValueError(f"{key} must be a nonempty identity")
    primary = result.get("primary_learned", variant in {"direct", "bounded_repair"})
    if not isinstance(primary, bool):
        raise ValueError("primary_learned must be boolean")
    if variant in {"direct", "bounded_repair", "exact_fallback"} and (
        guide != "learned" or not result.get("checkpoint_id")
    ):
        raise ValueError("learned controls require the same identified checkpoint and learned guide")
    if variant in {"bounded_repair", "prediction_free", "exact_fallback"} and not result.get("machinery_id"):
        raise ValueError("search controls require an identified decoder/repair machinery")
    if variant == "prediction_free" and (guide != "prediction_free" or result.get("checkpoint_id") or primary):
        raise ValueError("prediction_free control cannot use a learned guide or checkpoint")
    if (variant in {"exact_fallback", "oracle_diagnostic"} or guide == "oracle") and primary:
        raise ValueError("exact fallback and oracle diagnostics cannot be primary learned rows")
    if variant == "oracle_diagnostic" and guide != "oracle":
        raise ValueError("oracle_diagnostic must explicitly declare oracle assistance")
    if guide == "oracle" and variant != "oracle_diagnostic":
        raise ValueError("oracle assistance must be an explicitly diagnostic procedure")
    result["primary_learned"] = primary
    return result
