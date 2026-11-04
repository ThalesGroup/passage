"""Host-only check of the campaign-wide sealed scope before each Rule8 pass.

The launcher must establish and retain independent evidence that its plan was
sealed before the first pass. A digest or filesystem timestamp alone does not
establish that temporal fact. Version 1 carries one confirmatory analysis;
version 2 carries ``confirmatory_analyses`` keyed by orin, thor and tiger.
Each target/seed record still seals its own single-target analysis.
"""

from dataclasses import dataclass
from pathlib import Path

from passage_eval.io import pinned
from passage_eval.protocol.admission import AdmissionGrant, RULE8_NATIVE_PROFILE


@dataclass(frozen=True)
class Rule8Scope:
    path: Path
    sha256: str

    def check_record(self, grant, *, target=None):
        """Recheck both pins and match the all-GREEN record to this global plan."""
        if not isinstance(grant, AdmissionGrant) or not grant.is_genuine():
            raise ValueError("a genuine all-GREEN admission grant is required")
        grant.ensure_unchanged()
        plan = pinned.read_pinned_json(self.path, self.sha256)
        record = pinned.read_pinned_json(grant.record_path, grant.record_sha256)
        if not isinstance(plan, dict):
            raise ValueError("global Rule8 plan fields differ")
        schema = plan.get("schema")
        if schema == "passage-direct-rule8-global-scope/v1":
            if set(plan) != {"schema", "replica_catalog_sha256", "confirmatory_analysis"}:
                raise ValueError("global Rule8 plan fields differ")
            analysis = plan["confirmatory_analysis"]
        elif schema == "passage-direct-rule8-global-scope/v2":
            if set(plan) != {"schema", "replica_catalog_sha256", "confirmatory_analyses"}:
                raise ValueError("global Rule8 plan fields differ")
            analyses = plan["confirmatory_analyses"]
            if not isinstance(analyses, dict) or set(analyses) != {"orin", "thor", "tiger"}:
                raise ValueError("global Rule8 plan needs one analysis per target")
            from passage_eval.protocol.admission import decision_problem

            for device, value in analyses.items():
                if (not isinstance(value, dict) or value.get("target") != device
                        or decision_problem("confirmatory_analysis", value, {}, profile=RULE8_NATIVE_PROFILE)):
                    raise ValueError("global Rule8 target analysis differs")
            chosen = record.get("protocol", {}).get("decisions", {}).get("confirmatory_analysis", {}).get("value")
            chosen_target = chosen.get("target") if isinstance(chosen, dict) else None
            if (not isinstance(chosen_target, str) or chosen_target not in analyses
                    or (target is not None and target != chosen_target)):
                raise ValueError("record target differs from global Rule8 analysis")
            analysis = analyses[chosen_target]
        else:
            raise ValueError("global Rule8 plan schema differs")
        catalog = plan["replica_catalog_sha256"]
        if (
            not pinned.is_hex64(catalog)
            or not isinstance(analysis, dict)
            or analysis.get("seeds") != [1234, 2026, 3407]
        ):
            raise ValueError("global Rule8 catalogue or seeds differ")
        if target is not None and target != analysis.get("target"):
            raise ValueError("record target differs from global Rule8 analysis")
        if record["protocol"]["profile"] != RULE8_NATIVE_PROFILE:
            raise ValueError("Rule8 session requires an all-GREEN Rule8 grant")
        decisions = record["protocol"]["decisions"]
        seeds = decisions["seeds"]["value"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 1
            or seeds[0] not in analysis["seeds"]
            or decisions["confirmatory_analysis"]["value"] != analysis
        ):
            raise ValueError("Rule8 record analysis or single seed differs from global plan")
        _, catalog_sha = pinned.file_ref(record["evidence"]["replica_catalog"])
        if catalog_sha != catalog:
            raise ValueError("Rule8 record catalogue differs from global plan")
        grant.ensure_unchanged()
        return record


def load_rule8_scope(path, expected_sha256):
    """Pin the plan before the first session; callers must retain real ordering evidence."""
    pinned.read_pinned_json(path, expected_sha256)
    return Rule8Scope(Path(path), expected_sha256)
