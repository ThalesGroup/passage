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

"""Versioned supervised timing profiles; artifacts written without a profile keep their legacy meaning.

A profile fixes, together, the artifact schema, the timing boundary written in every record, the record
field that is compared with the wall budget and used as the latency, and what happens to the warm-up when a
worker is restarted inside a measured pass.

``supervised_total_v3`` (legacy, the default)
    ``total_wall_ms = service_wall_ms + validation_recost_ms`` is compared with the budget and scored, so the
    supervisor's validation and physical recost count against the candidate (boundary
    ``passage-timing-boundary/v3-supervised-ipc-fresh-pass``). It stays the default so that existing runs and
    receipts keep their meaning. The manuscript's timing rule keeps validation and recosting separate, which
    this profile does not.
``supervised_service_v1``
    ``service_wall_ms`` is compared and scored. Validation and recost are recorded in
    ``validation_recost_ms`` and kept in ``total_wall_ms``, outside the boundary
    (``passage-timing-boundary/v4-supervised-ipc-service``).
``supervised_service_rewarm_v1``
    As ``supervised_service_v1``; in addition, a worker restarted inside a warm pass is warmed up again, outside
    the boundary, before the next measured request, and every record states its warm-up state
    (``passage-timing-boundary/v5-supervised-ipc-service-rewarm``). The re-warm policy is a proposal for the
    protocol, not a sealed rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from passage_eval.supervision import REWARM_TIMING_BOUNDARY_ID, SERVICE_TIMING_BOUNDARY_ID

LEGACY_SCHEMA = "passage-complete-request-v1"
SERVICE_SCHEMA = "passage-complete-request-supervised-service-v2"
#: v2 records plus the per-request warm-up state of the re-warm profile.
REWARM_SCHEMA = "passage-complete-request-supervised-service-v3"
SERVICE_BOUNDARY = SERVICE_TIMING_BOUNDARY_ID
REWARM_BOUNDARY = REWARM_TIMING_BOUNDARY_ID

LEGACY = "supervised_total_v3"
SERVICE = "supervised_service_v1"
SERVICE_REWARM = "supervised_service_rewarm_v1"
NATIVE_PERSISTENT = "native_raw_path_persistent_v1"
NATIVE_SCHEMA = "passage-complete-request-native-v1"
NATIVE_BOUNDARY = "passage-timing-boundary/v6-native-raw-arrays-to-path"

#: Per-record fields of a re-warm profile: whether a re-warm ran just before the request, and whether the
#: workers that served it had completed a warm-up (``None`` where no warm-up applies, e.g. the cold series).
REWARM_RECORD_FIELDS = ("rewarmed_before_request", "warmup_complete_before_request")


@dataclass(frozen=True)
class TimingProfile:
    name: str
    schema: str
    #: ``None`` for the legacy profile, whose artifacts carry the boundary of their producer (in-process
    #: runner, supervised v2 or v3).
    boundary: str | None
    latency_field: str
    rewarm_after_restart: bool

    @property
    def validation_in_boundary(self) -> bool:
        return self.latency_field == "total_wall_ms"


_PROFILES = {
    LEGACY: TimingProfile(LEGACY, LEGACY_SCHEMA, None, "total_wall_ms", False),
    SERVICE: TimingProfile(SERVICE, SERVICE_SCHEMA, SERVICE_BOUNDARY, "service_wall_ms", False),
    SERVICE_REWARM: TimingProfile(SERVICE_REWARM, REWARM_SCHEMA, REWARM_BOUNDARY, "service_wall_ms", True),
    NATIVE_PERSISTENT: TimingProfile(NATIVE_PERSISTENT, NATIVE_SCHEMA, NATIVE_BOUNDARY, "native_wall_ms", False),
}
PROFILES = tuple(_PROFILES)
KNOWN_SCHEMAS = tuple(dict.fromkeys(profile.schema for profile in _PROFILES.values()))
PROFILED_SCHEMAS = tuple(profile.schema for profile in _PROFILES.values() if profile.boundary is not None)
PROFILED_BOUNDARIES = tuple(profile.boundary for profile in _PROFILES.values() if profile.boundary is not None)


def get_profile(name: str) -> TimingProfile:
    """The profile called ``name``; an unknown name is an error, never a fallback."""
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError(f"unknown timing profile {name!r}; known: {list(PROFILES)}")
    return _PROFILES[name]


def manifest_profile(manifest: Mapping) -> TimingProfile:
    """The timing profile of a run manifest, checked against its procedure, schema and boundary."""
    name = manifest.get("timing_profile", LEGACY)
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError(f"unknown timing profile {name!r}")
    profile = _PROFILES[name]
    procedure = manifest.get("procedure", {})
    if not isinstance(procedure, Mapping):
        raise ValueError("procedure must be a mapping")
    section = "execution" if name == NATIVE_PERSISTENT else "supervision"
    execution = procedure.get(section, {})
    if not isinstance(execution, Mapping):
        raise ValueError(f"{section} must be a mapping")
    if execution.get("timing_profile", LEGACY) != name:
        raise ValueError(f"timing profile must match the {section} procedure")
    if profile.boundary is None:
        if manifest.get("schema") in PROFILED_SCHEMAS or manifest.get("timing_boundary") in PROFILED_BOUNDARIES:
            raise ValueError("a profiled schema or timing boundary requires its timing profile")
    elif manifest.get("schema") != profile.schema or manifest.get("timing_boundary") != profile.boundary:
        raise ValueError(f"timing profile {name} requires its versioned schema and boundary")
    return profile


def latency_field(manifest: Mapping) -> str:
    """The record field that a manifest's profile compares with the budget and scores as latency."""
    return manifest_profile(manifest).latency_field


WARMUP_EVIDENCE_SCHEMA = "passage-latency-warmup-evidence/v1"


def _count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def manifest_warmup(manifest: Mapping) -> dict:
    """The warm-up that a run manifest declares, and records as performed, before its timed series.

    The series name alone says nothing about a warm-up: both runners call their measured series ``warm`` even
    when no warm-up ran (``measure_latency`` false or ``latency_warmup`` zero). This reads the evidence instead.

    * Supervised runs: ``supervision.timing_policy`` declares ``latency_warmup`` untimed requests per warm pass
      on inputs disjoint from the roster; each entry of ``supervision.passes`` records the warm-up summary of its
      pass. The warm-up is *performed* when every pass of the series ran at least one warm-up request.
    * In-process ``cli`` runs: the ``bench_spec`` declares ``latency_warmup`` recorded passes over the roster when
      ``measure_latency`` is true; the CLI runner runs them before its warm series, and the warm artifact does not
      list them, so a declared warm-up is taken as performed.
    * Any other manifest declares no warm-up.
    """
    series = manifest.get("series")
    if manifest.get("timing_profile") == NATIVE_PERSISTENT:
        execution = manifest.get("native_execution")
        execution = execution if isinstance(execution, Mapping) else {}
        warmup = execution.get("warmup")
        warmup = warmup if isinstance(warmup, Mapping) else {}
        calls = _count(warmup.get("k")) * _count(warmup.get("repeats"))
        declared = series == "warm" and calls > 0
        digest = warmup.get("input_roster_sha256")
        performed = (
            declared
            and warmup.get("k") == 1
            and warmup.get("repeats") == 16
            and warmup.get("performed") is True
            and warmup.get("disjoint_by_content") is True
            and isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
            and type(execution.get("n_processes")) is int
            and execution["n_processes"] == 1
            and type(execution.get("n_restarts")) is int
            and execution["n_restarts"] == 0
            and isinstance(execution.get("session_id"), str)
            and bool(execution["session_id"].strip())
        )
        return {
            "schema": WARMUP_EVIDENCE_SCHEMA,
            "source": "native_initial_warmup",
            "unit": "untimed_calls_on_one_disjoint_validation_input_per_persistent_process",
            "declared": declared,
            "performed": performed,
            "per_pass": calls,
            "passes": 1,
            "passes_with_warmup": int(performed),
            "passes_with_complete_warmup": int(performed),
        }
    supervision = manifest.get("supervision")
    if isinstance(supervision, Mapping) and isinstance(supervision.get("timing_policy"), Mapping):
        policy = supervision["timing_policy"]
        source = policy.get("warmup")
        per_pass = _count(policy.get("latency_warmup"))
        declared = (
            series == "warm"
            and per_pass > 0
            and isinstance(source, Mapping)
            and source.get("policy") not in (None, "none")
        )
        passes = [
            entry
            for entry in supervision.get("passes", ())
            if isinstance(entry, Mapping) and entry.get("series") == series
        ]
        summaries = [entry.get("warmup") for entry in passes]
        ran = [summary for summary in summaries if isinstance(summary, Mapping) and _count(summary.get("requests"))]
        # Summaries written before completeness was recorded leave the count unknown (None).
        recorded = all(
            summary is None or (isinstance(summary, Mapping) and "complete" in summary) for summary in summaries
        )
        return {
            "schema": WARMUP_EVIDENCE_SCHEMA,
            "source": "supervised_timing_policy_and_pass_log",
            "unit": "untimed_requests_per_warm_pass_on_disjoint_inputs",
            "declared": declared,
            "per_pass": per_pass,
            "passes": len(passes),
            "passes_with_warmup": len(ran),
            "passes_with_complete_warmup": (
                sum(isinstance(summary, Mapping) and summary.get("complete") is True for summary in summaries)
                if passes and recorded
                else None
            ),
            "performed": declared and bool(passes) and len(ran) == len(passes),
        }
    bench = manifest.get("bench_spec")
    if isinstance(bench, Mapping):
        per_pass = _count(bench.get("latency_warmup")) if bench.get("measure_latency") is True else 0
        declared = series == "warm" and per_pass > 0
        return {
            "schema": WARMUP_EVIDENCE_SCHEMA,
            "source": "bench_spec_in_process_runner",
            "unit": "recorded_warmup_passes_over_the_roster_before_the_warm_series",
            "declared": declared,
            "per_pass": per_pass,
            "passes": None,
            "passes_with_warmup": None,
            "passes_with_complete_warmup": None,
            "performed": declared,
        }
    return {
        "schema": WARMUP_EVIDENCE_SCHEMA,
        "source": "none",
        "unit": None,
        "declared": False,
        "per_pass": 0,
        "passes": None,
        "passes_with_warmup": None,
        "passes_with_complete_warmup": None,
        "performed": False,
    }
