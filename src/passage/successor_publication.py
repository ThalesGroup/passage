"""Receipt-derived publication files for the released dataset; source documents stay intact."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import pyarrow.parquet as pq
import yaml

from passage.notebook_export import write_staged_bytes

if TYPE_CHECKING:
    from passage.export import ReleaseContext

VERSION = "passage-successor-v1"
PUBLIC_VERSION = "1.0.0"
RECEIPT = "release-receipts/publication-metadata.json"
FACTS = "release-receipts/publication-facts.json"
SPLITS = ("train", "calibration", "validation", "test")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _facts(root: Path) -> tuple[dict, dict[str, str]]:
    """Check scientific identities, counts and source controls before drafting claims."""
    verification = _json(root / "release-receipts/release-verification.json")
    if (
        verification.get("dataset_version") != VERSION
        or verification.get("data_status") != "verified"
        or verification.get("status") != "verified"
        or verification.get("admission", {}).get("status") != "qualified"
        or verification["admission"].get("unresolved_variants") != 0
    ):
        raise ValueError("Release publication requires a verified, qualified release without unresolved variants")
    from passage.calibrate import validate_calibration_receipt
    from passage.regenerate import _split_counts
    from passage.successor_generation import semantic_hash
    from passage_eval.protocol.contracts import GraphSpec

    controls = [
        "release-receipts/release-verification.json",
        "release-receipts/resolved-config.yaml",
        "release-receipts/calibration.json",
        "release-receipts/generation-provenance.json",
        "release-receipts/producer-source.zip",
    ]
    if (root / "notebook-export-manifest.json").is_file():
        controls.append("notebook-export-manifest.json")
    config = yaml.safe_load((root / controls[1]).read_text())
    calibration = _json(root / controls[2])
    low, high = validate_calibration_receipt(calibration)
    provenance = _json(root / controls[3])
    provenance_id = provenance.get("provenance_id")
    if (
        provenance.get("dataset_version") != VERSION
        or config.get("generation", {}).get("dataset_version") != VERSION
        or provenance.get("config_sha256") != _hash(root / controls[1])
        or provenance.get("calibration_sha256") != _hash(root / controls[2])
        or provenance.get("configuration") != json.loads(json.dumps(config))
        or semantic_hash(
            {
                **{k: v for k, v in provenance.items() if k not in ("provenance_id", "configuration")},
                "configuration": config,
            }
        )
        != provenance_id
    ):
        raise ValueError("Release publication generation/configuration/calibration identity mismatch")
    if provenance.get("nodata_policy") != "reject-source-crop":
        raise ValueError("Unsupported release nodata policy")
    sources = {item["name"]: item["sha256"] for item in provenance["sources"].values()}
    calibrated_sources = {item["filename"]: item["sha256"] for item in calibration["source_inventory"]}
    if sources != calibrated_sources:
        raise ValueError("Release publication calibration/source inventory mismatch")
    with zipfile.ZipFile(root / controls[4]) as archive:
        for path, expected in provenance["source_snapshot"].items():
            if hashlib.sha256(archive.read(f"src/{path}")).hexdigest() != expected:
                raise ValueError("Release publication archived producer differs from generating source")
    from passage.export import _features

    graphs = {
        item["name"]: GraphSpec(cost_model=item["cost_model"], cost_weight=item["weight"]).to_dict()
        for item in config["solvers"]
    }
    if len(graphs) != 3:
        raise ValueError("Release publication requires three objective graphs")
    resolutions, configurations = {}, []
    for key, summary in sorted(verification["resolutions"].items(), key=lambda item: int(item[0])):
        resolution = int(key)
        configured = config["num_samples"]
        expected_total = (
            int(configured.get(resolution, configured.get(key, 0))) if isinstance(configured, dict) else int(configured)
        )
        if verification.get("scope") == "full" and (
            summary["requested_scenes"] != expected_total
            or summary["core_counts"] != _split_counts(expected_total, config["splits"]["ratios"])
        ):
            raise ValueError("Release publication core counts differ from configured full population")
        if (
            summary["unresolved_variants"] != 0
            or summary["successful_route_variants"] + summary["proved_infeasible_variants"]
            != 6 * summary["requested_scenes"]
        ):
            raise ValueError("Release publication reference/terminal denominator mismatch")
        for suffix, count_key in (("", "core_counts"), ("_samples", "sample_counts")):
            name = f"{resolution}x{resolution}{suffix}"
            configurations.append(name)
            for split in SPLITS:
                shards = sorted((root / name / split).glob("*.parquet"))
                count = 0
                if not shards:
                    raise ValueError(f"Missing release publication split: {name}/{split}")
                for shard in shards:
                    with pq.ParquetFile(shard) as parquet:
                        count += parquet.metadata.num_rows
                        if not parquet.schema_arrow.equals(
                            _features(bool(suffix), list(graphs)).arrow_schema, check_metadata=False
                        ):
                            raise ValueError("Release publication Arrow schema differs from declared card features")
                        images = [column for column in parquet.schema_arrow.names if column.startswith("image_")]
                        if bool(images) != bool(suffix):
                            raise ValueError("Release publication core/preview image schema mismatch")
                if count != summary[count_key].get(split):
                    raise ValueError("Release publication footer count differs from verification")
                if (
                    suffix
                    and verification.get("scope") == "full"
                    and count != min(64, int(config["export"]["sample"]["num_samples"]))
                ):
                    raise ValueError("Release publication preview count differs from configured sample size")
        state_relative = f"release-receipts/state-{resolution}.json"
        state = _json(root / state_relative)
        if (
            state.get("generation_provenance_id") != provenance_id
            or state.get("config_sha256") != provenance["config_sha256"]
            or state.get("calibration_sha256") != provenance["calibration_sha256"]
        ):
            raise ValueError("Release publication continuation provenance mismatch")
        last_relative = f"release-receipts/{state['last_manifest']}"
        if _hash(root / last_relative) != state["last_manifest_sha256"]:
            raise ValueError("Release publication final batch manifest mismatch")
        controls.extend([state_relative, last_relative])
        for prefix in ("cohort", "supervision"):
            relative = f"release-receipts/{prefix}-{resolution}.jsonl"
            if _hash(root / relative) != summary[f"{prefix}_sha256"]:
                raise ValueError("Release publication qualified cohort identity mismatch")
            controls.append(relative)
        resolutions[key] = summary
    if not resolutions:
        raise ValueError("Release publication requires verified resolution populations")
    endpoint_policy, repair = provenance["endpoint_policy"], None
    repair_relative = "release-receipts/connectivity-repair-v1.json"
    if (root / repair_relative).is_file():
        repair = _connectivity_repair(root, repair_relative, resolutions)
        endpoint_policy = repair["endpoint_policy"]
        controls.extend([repair_relative, repair["source_archive"]])
    facts = {
        "dataset_version": VERSION,
        "scope": verification["scope"],
        "configurations": configurations,
        "resolutions": resolutions,
        "core_scenes": sum(s["requested_scenes"] for s in resolutions.values()),
        "preview_rows": sum(sum(s["sample_counts"].values()) for s in resolutions.values()),
        "route_variants": sum(s["requested_scenes"] * 6 for s in resolutions.values()),
        "successful_route_variants": sum(s["successful_route_variants"] for s in resolutions.values()),
        "proved_infeasible_variants": sum(s["proved_infeasible_variants"] for s in resolutions.values()),
        "calibration": {
            "min_m": low,
            "max_m": high,
            "sha256": provenance["calibration_sha256"],
            "source_inventory_sha256": calibration["source_inventory_sha256"],
        },
        "generation_provenance_id": provenance_id,
        "graphs": graphs,
        "geographic_policy": provenance["geographic_policy"],
        "nodata_policy": provenance["nodata_policy"],
        "terrain_normalization": provenance["terrain_normalization"],
        "sample_seeding": provenance["sample_seeding"],
        "tie_policy": provenance["tie_policy"],
        "scene_attempts": provenance["scene_attempts"],
        "endpoint_policy": endpoint_policy,
        "split_ratios": config["splits"]["ratios"],
        "split_seed": config["splits"]["seed"],
        "learned_results": "not_run",
        "upload": "not_run",
    }
    if repair is not None:
        facts["connectivity_repair"] = {
            key: repair[key] for key in ("repaired_scenes", "generation_provenance_id", "code_commit")
        }
    return facts, {path: _hash(root / path) for path in controls}


def _connectivity_repair(root: Path, relative: str, resolutions: dict) -> dict:
    """Check the receipt of scenes regenerated under the disconnected-endpoint rule.

    Each listed scene must sit in its batch manifest with the new receipt, every replaced
    shard must carry its new hash there, and the archived generator must be the recorded one.
    """
    receipt = _json(root / relative)
    generator = receipt["generator"]
    if receipt.get("schema") != "passage-connectivity-repair/v1" or receipt["repaired_scenes"] != len(
        receipt["scenes"]
    ):
        raise ValueError("Connectivity repair receipt identity mismatch")
    archive_path = root / generator["source_archive"]
    if _hash(archive_path) != receipt["source_archive_sha256"]:
        raise ValueError("Connectivity repair source archive mismatch")
    from passage.successor_generation import semantic_hash

    with zipfile.ZipFile(archive_path) as archive:
        snapshot = {
            name.removeprefix("src/"): hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
        }
    if semantic_hash(snapshot) != generator["source_snapshot_sha256"]:
        raise ValueError("Connectivity repair archived generator differs from the recorded source")
    by_batch = {}
    for scene in receipt["scenes"]:
        by_batch.setdefault(scene["batch"], []).append(scene)
    batch_files = {}
    for path in sorted((root / "release-receipts").glob("batch-*.json")):
        manifest = _json(path)
        for record in manifest["files"]:
            batch_files[record["path"]] = record["sha256"]
        if path.name not in by_batch:
            continue
        committed = {(s["split"], s["index"]): s for s in manifest["scenes"]}
        for scene in by_batch.pop(path.name):
            found = committed.get((scene["split"], scene["index"]))
            if (
                found is None
                or found["receipt_semantic_sha256"] != scene["new_receipt_semantic_sha256"]
                or found["receipt"]["provenance_id"] != generator["provenance_id"]
                or any(v["status"] != "ok" for v in found["receipt"]["variants"].values())
            ):
                raise ValueError(f"Connectivity repair scene not committed: {scene['resolution']}/{scene['index']}")
    if by_batch:
        raise ValueError("Connectivity repair lists a batch manifest that is not published")
    if any(batch_files.get(shard["path"]) != shard["new"]["sha256"] for shard in receipt["shards"]):
        raise ValueError("Connectivity repair shard hash differs from its batch manifest")
    if sum(s["proved_infeasible_variants"] for s in resolutions.values()) != 0:
        raise ValueError("Connectivity repair leaves variants without a path")
    return {
        "repaired_scenes": receipt["repaired_scenes"],
        "generation_provenance_id": generator["provenance_id"],
        "code_commit": generator["code_commit"],
        "endpoint_policy": generator["endpoint_policy"],
        "source_archive": generator["source_archive"],
    }


def labels_paragraph(facts: dict) -> str:
    """Label-status paragraph of the generated card and datasheet."""
    if facts["proved_infeasible_variants"] == 0:
        return (
            "Every route variant has status `ok`: a legal, exact route under the frozen GraphSpec. Start and goal "
            "cells that the obstacles disconnect are rejected by the generator. Failed/censored outcomes would "
            "retain explicit terminal records and block qualification. Supervision rosters list every variant. "
            "Tied optima may have different path geometry.\n\n"
        )
    return (
        "Successful `ok` paths are legal and exact under the frozen GraphSpec. `infeasible` means the reference solve "
        "proved no path; its ordered route is empty and cost is null. Failed/censored outcomes retain explicit terminal "
        "records and block qualification. Successful-path supervision uses separate supervision rosters; infeasible "
        "tasks do not become zero-cost successful examples. Tied optima may have different path geometry.\n\n"
    )


def _description(facts: dict) -> str:
    # The published card, datasheet and Croissant metadata embed this exact text, and
    # verification recomputes it.
    if facts["proved_infeasible_variants"] == 0:
        outcomes = (
            f"All {facts['successful_route_variants']:,} variants are exact routes; start and goal cells "
            "that the obstacles disconnect are rejected by the generator."
        )
    else:
        outcomes = (
            f"The qualified outcomes include {facts['successful_route_variants']:,} successful exact routes "
            f"and {facts['proved_infeasible_variants']:,} independently proved no-path variants."
        )
    return (
        f"PASSAGE v1.0 contains {facts['core_scenes']:,} core scenes and "
        f"{facts['route_variants']:,} objective/obstacle variants; {facts['preview_rows']:,} separate preview rows "
        "do not add independent training observations. Terrain and labels use persisted float32 inputs decoded "
        f"before authoritative float64 graph solves. Missing/nodata source crops are rejected. {outcomes}"
    )


def _repository_ref(repo_id: str, revision: str) -> tuple[str, str]:
    """Return a Croissant-compatible Git URL and the corresponding Hub revision."""
    if (
        not isinstance(revision, str)
        or not revision
        or ".." in revision
        or re.fullmatch(r"[A-Za-z0-9_./-]+", revision) is None
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision)
    ):
        raise ValueError("Hugging Face revision must be main, a tag, or an explicit refs/heads or refs/tags ref")
    repository = f"https://huggingface.co/datasets/{repo_id}"
    if revision == "main":
        return repository, revision
    ref = revision if revision.startswith(("refs/heads/", "refs/tags/")) else f"refs/tags/{revision}"
    if ref.endswith("/") or revision.startswith("refs/") and ref != revision:
        raise ValueError("Unsupported Hugging Face revision ref")
    return f"{repository}/tree/{quote(ref, safe='')}", ref.split("/", 2)[2]


def _publication_responsibility(facts: dict, calibration: dict, revision: str) -> dict:
    """Build every RAI/PROV assertion from current facts or explicit project context."""
    source_count = len(calibration["source_inventory"])
    elevation_range = f"{facts['calibration']['min_m']!r} m to {facts['calibration']['max_m']!r} m"
    labels = labels_paragraph(facts).strip().replace("`", "")
    geography = json.dumps(facts["geographic_policy"], sort_keys=True)
    graphs = json.dumps(facts["graphs"], sort_keys=True)
    grids = ", ".join(f"{size}x{size}" for size in sorted(facts["resolutions"], key=int))
    objectives = ", ".join(sorted({graph["cost_model"] for graph in facts["graphs"].values()}))
    metadata = {
        "rai:dataCollection": (
            "Observational terrain from the JAXA ALOS World 3D-30m (AW3D30) digital surface model, with "
            "procedural obstacles, start/goal markers and computed reference routes. "
            f"The frozen calibration inventory pins {source_count:,} source tiles by filename, size and SHA-256 "
            "in calibration/calibrate.json. No human subjects, crowdworkers or manual labelling."
        ),
        "rai:dataCollectionType": [
            "Secondary data analysis (JAXA AW3D30 digital surface model)",
            "Software collection (procedural obstacles, markers and reference routes)",
        ],
        "rai:dataCollectionRawData": (
            "Terrain source files are identified in calibration/calibrate.json. Recorded terrain normalization: "
            f"{facts['terrain_normalization']}. Raster dtype, tile dimensions and product versions are not inferred "
            "from filenames. Raw terrain files are separate from the exported scene configurations."
        ),
        "rai:dataCollectionMissingData": (
            f"Recorded nodata policy: {facts['nodata_policy']}. Missing or invalid source crops are rejected. "
            f"Maximum scene attempts: {facts['scene_attempts']}; endpoint policy: {facts['endpoint_policy']}. "
            f"The qualified population contains {facts['core_scenes']:,} core scenes. {labels}"
        ),
        "rai:dataImputationProtocol": (
            f"Recorded nodata policy: {facts['nodata_policy']}. Rejected source crops are not imputed, padded or clipped."
        ),
        "rai:dataPreprocessingProtocol": [
            f"Calibrate the elevation range over the authenticated source inventory: {elevation_range}.",
            f"Assign source tiles and complete crop support using policy {geography}; split seed {facts['split_seed']}.",
            f"Per-scene seed derivation: {facts['sample_seeding']}; terrain normalization: {facts['terrain_normalization']}.",
            f"Generate obstacles and distinct markers with endpoint policy {facts['endpoint_policy']}.",
            "Store the float32 elevation/markers/obstacles tensor; decode it before all six float64 reference solves.",
            f"Export {len(facts['configurations'])} Parquet configurations ({grids}) with structured metadata, "
            "JSON replay metadata and scene receipts; previews add rendered PNG images.",
        ],
        "rai:dataAnnotationProtocol": (
            "Computational exact float64 reference solves of all six variants from decoded persisted float32 "
            f"inputs under these frozen GraphSpec contracts: {graphs}. {labels}"
        ),
        "rai:dataAnnotationPlatform": "Computational annotation by the PASSAGE reference solver; no human labelling platform.",
        "rai:dataAnnotationAnalysis": (
            "Release verification is verified and admission is qualified with zero unresolved variants: "
            f"{facts['successful_route_variants']:,} exact routes and {facts['proved_infeasible_variants']:,} "
            "proved infeasible variants. Cohort and supervision identities are in release-receipts/publication-facts.json. "
            f"Tie policy: {facts['tie_policy']}. Population verification does not establish independent replay of "
            "every route or cross-platform reproducibility."
        ),
        "rai:annotationsPerItem": (
            f"Six route variants per core scene, {facts['route_variants']:,} in total; they share terrain and endpoints "
            "and are not independent observations. One deterministic reference solve per variant; no multi-annotator redundancy."
        ),
        "rai:annotatorDemographics": "Not applicable: no human annotators.",
        "rai:machineAnnotationTools": [
            "PASSAGE generator (src/passage/successor_generation.py)",
            "Exact float64 reference solver (src/passage/reference.py) under the passage-eval GraphSpec contract",
            f"Per-scene seed derivation: {facts['sample_seeding']}",
        ],
        "rai:dataBiases": [
            f"Geographic coverage is restricted to {source_count:,} calibrated source tiles and admitted crops. "
            f"The recorded split and guard policy is {geography}.",
            f"Elevation normalization uses the calibrated range {elevation_range}. "
            "The inventory alone does not establish ground-area-weighted or operationally representative sampling.",
            "Procedural endpoints and obstacles follow the recorded generation policy; representativeness of "
            "operational missions is not established.",
            "Digital surface model elevations can include canopy and building surfaces, affecting elevation-based routing proxies.",
        ],
        "rai:dataLimitations": [
            "Static grid tasks with procedural obstacles and endpoints do not model vehicle dynamics, clearance, "
            "weather, airspace, regulation or mission demand.",
            "Reference labels are exact under the recorded GraphSpec contracts on decoded float32 inputs. "
            "Other graph conventions change optima; tied optima can differ in geometry.",
            f"Six variants share each scene's terrain and endpoints. The {facts['preview_rows']:,} preview rows "
            "duplicate admitted scenes; correlated observations should not be treated as independent samples.",
            "This research benchmark does not certify operational flight planning, search-and-rescue dispatch "
            "or other safety-of-life deployment.",
        ],
        "rai:dataUseCases": [
            f"Benchmark terrain-aware path planners against the released GraphSpec objectives: {objectives}.",
            "Train and evaluate routing models on the fixed train/calibration/validation/test splits.",
            f"Inspect performance at the released grid sizes: {grids}. Geographic evaluation follows policy {geography}.",
        ],
        "rai:dataSocialImpact": (
            "PASSAGE supports research on verifiable terrain-aware planners. Over-trusting benchmark performance "
            "for operational deployment is a risk; exact references, explicit failure accounting and the stated "
            "graph contract support evaluation. The dataset does not certify a navigation product."
        ),
        "rai:personalSensitiveInformation": (
            "The data contain JAXA elevation, procedural obstacles and markers, and computed reference routes. "
            "No human subjects are collected. Privacy review and redaction of artifact metadata are not established "
            "by this metadata."
        ),
        "rai:hasSyntheticData": True,
        "rai:dataReleaseMaintenancePlan": (
            "PASSAGE v1.0 release identities and metadata are recorded in release-receipts/; "
            f"this metadata targets Hugging Face revision {revision}. Maintainers are the authors at Thales; contact "
            "through https://github.com/ThalesGroup/passage/issues or geoffrey.delhomme@thalesgroup.com."
        ),
        "prov:wasDerivedFrom": [
            {
                "@id": "https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm",
                "prov:label": "JAXA ALOS Global Digital Surface Model 'ALOS World 3D - 30m' (AW3D30)",
                "sc:description": (
                    "AW3D30 digital surface model from the ALOS PRISM stereo sensor. "
                    f"This release uses {source_count:,} calibrated source tiles whose filenames, sizes and SHA-256 "
                    "identities are recorded in calibration/calibrate.json. Tile versions are not inferred from "
                    "filenames. Terrain is observational; obstacles, endpoints and routes are generated."
                ),
                "sc:license": "https://earth.jaxa.jp/en/data/policy/",
                "prov:wasAttributedTo": {
                    "@id": "https://global.jaxa.jp/",
                    "prov:label": "Japan Aerospace Exploration Agency (JAXA)",
                },
            }
        ],
    }
    activities = [
        (
            "Source inventory and calibration",
            f"Source inventory and calibration ({elevation_range}) are pinned in calibration/calibrate.json.",
            "calibrate.py",
        ),
        (
            "Scene generation",
            f"Generation provenance id {facts['generation_provenance_id']}; geographic policy "
            f"{geography}. Configuration and producer source are archived in release-receipts/resolved-config.yaml "
            "and release-receipts/producer-source.zip.",
            "successor_generation.py",
        ),
        ("Reference route annotation", labels, "reference.py"),
        (
            "Export, verification and publication admission",
            "Parquet export and release admission are bound to "
            "release-receipts/release-verification.json and release-receipts/publication-facts.json. Executed notebooks, "
            "when present, are bound in notebook-export-manifest.json.",
            "export.py",
        ),
    ]
    metadata["prov:wasGeneratedBy"] = [
        {
            "@type": "prov:Activity",
            "prov:label": label,
            "sc:description": description,
            "prov:wasAttributedTo": [
                {
                    "@type": "prov:SoftwareAgent",
                    "prov:label": f"PASSAGE {label}",
                    "@id": f"https://github.com/ThalesGroup/passage/blob/main/src/passage/{filename}",
                }
            ],
        }
        for label, description, filename in activities
    ]
    return metadata


def _publication_croissant(source: dict, facts: dict, root: Path, repo_id: str, revision: str) -> dict:
    """Retain the source schema and RAI coverage; bind population claims to receipts."""
    container_url, hub_revision = _repository_ref(repo_id, revision)
    repository = f"https://huggingface.co/datasets/{repo_id}"
    now = datetime.now(timezone.utc).isoformat()
    source.update(
        {
            "@id": repository,
            "url": repository,
            "version": PUBLIC_VERSION,
            "description": _description(facts),
            "dateModified": now,
        }
    )
    source.setdefault("datePublished", now)
    licenses = source["license"] if isinstance(source["license"], list) else [source["license"]]
    source["license"] = [
        f"{repository}/blob/{quote(hub_revision, safe='')}/DATASET_LICENSE.md" if "/DATASET_LICENSE.md" in url else url
        for url in licenses
    ]
    configurations = set(facts["configurations"])
    source["recordSet"] = [record for record in source["recordSet"] if record["name"] in configurations]
    if {record["name"] for record in source["recordSet"]} != configurations:
        raise ValueError("Source Croissant schema does not cover the verified release configurations")
    from passage.export import _features

    for record in source["recordSet"]:
        expected = set(_features(record["name"].endswith("_samples"), list(facts["graphs"])))
        if {field["name"] for field in record["field"]} != expected:
            raise ValueError("Source Croissant schema columns differ from the verified release")
    filesets = {
        field["source"]["fileSet"]["@id"]
        for record in source["recordSet"]
        for field in record["field"]
        if "source" in field
    }
    distributions = {item["@id"]: item for item in source["distribution"]}
    containers = {distributions[name]["containedIn"]["@id"] for name in filesets}
    source["distribution"] = [item for item in source["distribution"] if item["@id"] in filesets | containers]
    for item in source["distribution"]:
        if item["@id"] in containers:
            item["contentUrl"] = container_url
            item["description"] = (
                f"Hugging Face dataset repository {repo_id}, revision {revision}. "
                "The sha256 value is the documented Hugging Face git-container sentinel, not a content digest; "
                "per-file identities are recorded in release-receipts/."
            )
        else:
            name = item["name"]
            summary = facts["resolutions"][name.split("x", 1)[0]]
            counts = summary["sample_counts" if name.endswith("_samples") else "core_counts"]
            population = ", ".join(f"{split} {counts[split]:,}" for split in SPLITS)
            item["description"] = f"Parquet shards of configuration {name}, one folder per split ({population} rows)."
    for record in source["recordSet"]:
        if record["name"].endswith("_samples"):
            record["description"] = (
                "Preview rows with rendered images; split counts are in the corresponding "
                "FileSet. Inspection examples duplicate admitted scenes and add no training observations."
            )
        if facts["proved_infeasible_variants"]:
            for field in record["field"]:
                if field["name"].startswith("path_"):
                    field["description"] = (
                        "Ordered [row, col] cells of the exact reference route, start to goal. "
                        "The corresponding metadata.paths terminal status distinguishes ok routes from proved "
                        "infeasible variants, whose ordered route is empty and cost is null."
                    )
    responsibility = _publication_responsibility(facts, _json(root / "release-receipts/calibration.json"), revision)
    inherited_keys = {key for key in source if key.startswith(("rai:", "prov:"))}
    if inherited_keys - responsibility.keys():
        raise ValueError(f"Unbound responsibility metadata: {sorted(inherited_keys - responsibility.keys())}")
    # Templates provide schema and project identity, never publication-specific assertions.
    source.update(responsibility)
    return source


def write_successor_publication_metadata(
    context: ReleaseContext, *, repo_id: str = "thalesgroup/passage", revision: str = "main"
) -> dict:
    """Write receipt-bound metadata, optionally targeting a Hub tag or explicit Git ref.

    ``revision`` defaults to the repository root (main). Bare names identify tags;
    branches use ``refs/heads/name``. Croissant's Git loader requires encoded refs
    and cannot load a bare commit SHA; use a tag for an immutable release identity.
    """
    _repository_ref(repo_id, revision)
    root = context.export_dirpath.resolve()
    facts, inputs = _facts(root)
    if (
        _hash(context.config_file) != inputs["release-receipts/resolved-config.yaml"]
        or _hash(context.calibration_file) != facts["calibration"]["sha256"]
    ):
        raise ValueError("Release publication current controls differ from archived inputs")
    from passage.export import _generate_huggingface_yaml_header, _repo_root

    source_root = _repo_root()
    description = _description(facts)
    rows = ["| Resolution | Split | Core scenes | Preview scenes |", "| --- | --- | ---: | ---: |"]
    for resolution, summary in facts["resolutions"].items():
        for split in SPLITS:
            rows.append(
                f"| {resolution} | {split} | {summary['core_counts'][split]:,} | {summary['sample_counts'][split]:,} |"
            )
    evidence = (
        "[release verification](release-receipts/release-verification.json), "
        "[generation provenance](release-receipts/generation-provenance.json), "
        "[publication facts](release-receipts/publication-facts.json) and "
        "[metadata hashes](release-receipts/publication-metadata.json)"
    )
    body = (
        f"# PASSAGE v1.0\n\n{description}\n\n"
        "## Population and inputs\n\n" + "\n".join(rows) + "\n\n"
        f"The complete calibration range is **{facts['calibration']['min_m']!r} m to {facts['calibration']['max_m']!r} m**. "
        "See [calibrate.json](calibration/calibrate.json) for the authenticated source inventory and validity policy. "
        "The float32 tensor stores normalized elevation, start/goal markers and a binary obstacle channel. "
        "Core Parquets contain no image columns. Previews depict the lowest admitted complete scene IDs per split; "
        "they are inspection examples rather than population evidence. Full replay metadata is retained in `metadata_json` "
        "and `scene_receipt`; ordinary graph parameters and terminal path statuses are also structured metadata.\n\n"
        "## Graph and label contract\n\n"
        "The routing graph is eight-connected with destination-only diagonal traversal, blocked obstacles and a fixed "
        "30 m graph distance convention. Elevation/slope/energy costs are routing proxies on the JAXA ALOS AW3D30 "
        "digital surface model, which includes canopy/building surfaces; 30 m graph spacing does not certify actual "
        "ground spacing. Calibration and objective weights remain float64 in exported metadata.\n\n"
        + labels_paragraph(facts)
        + "```json\n"
        + json.dumps(facts["graphs"], indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Geography and replay\n\n"
        "Source-cell windows, including neighboring tiles and wrapped longitude, are retained in scene receipts. "
        "The frozen source tile partition and guard policy is recorded below; complete crop support must belong "
        "to its assigned split. Missing tiles, nodata and incomplete source windows are rejected rather than padded.\n\n"
        "```json\n" + json.dumps(facts["geographic_policy"], indent=2, sort_keys=True) + "\n```\n\n"
        f"Sample seeding: `{facts['sample_seeding']}`. Normalization: `{facts['terrain_normalization']}`. "
        f"Tie policy: `{facts['tie_policy']}`. Maximum scene attempts: {facts['scene_attempts']}. "
        f"Endpoint policy: `{facts['endpoint_policy']}`. All requested scene/variant outcomes remain in release receipts.\n\n"
        "Examples are the verified `_samples` Parquets and executed display notebooks.\n\n"
        "## Reproduction and results\n\n"
        "Use an explicit local published dataset root, or explicitly select Hugging Face with the actual published "
        "40-hex commit revision. A local missing file never causes network fallback. "
        "Executed notebooks and their artifacts are bound in "
        "[notebook-export-manifest.json](notebook-export-manifest.json); portable parameter edits are recorded. "
        "Dataset qualification covers the released inputs and reference labels. Model evaluations require "
        "identified checkpoints, protocol configurations and measured result receipts.\n\n"
        f"This card derives its scientific claims from {evidence}.\n\n"
        "## Licensing and citation\n\n"
        "Generator code uses the [MIT license](LICENSE). Derived dataset artifacts retain JAXA ALOS AW3D30 "
        "attribution and upstream usage obligations in [Dataset Terms](DATASET_LICENSE.md). "
        "See [CITATION.cff](CITATION.cff) and [DATASHEET.md](DATASHEET.md).\n"
    )
    citation = yaml.safe_load((source_root / "CITATION.cff").read_text())
    citation.update(
        version=PUBLIC_VERSION,
        url=f"https://huggingface.co/datasets/{repo_id}",
        **{"repository-artifact": f"https://huggingface.co/datasets/{repo_id}"},
    )
    citation.pop("date-released", None)
    citation_text = yaml.safe_dump(citation, sort_keys=False)
    croissant = _publication_croissant(_json(source_root / "croissant.json"), facts, root, repo_id, revision)
    default = (
        context.configured_default
        if context.configured_default in facts["configurations"]
        else next(name for name in facts["configurations"] if name.endswith("_samples"))
    )
    header = _generate_huggingface_yaml_header(
        repo_id, facts["configurations"], list(SPLITS), default, list(facts["graphs"])
    )
    outputs = {
        "README.md": (header + body).encode(),
        "DATASHEET.md": (
            "# PASSAGE v1.0 datasheet\n\n" + description + "\n\n"
            "This datasheet documents the computational annotation and qualification process, not model results or "
            "an operational deployment certification.\n\n" + body
        ).encode(),
        "croissant.json": (json.dumps(croissant, indent=2, sort_keys=True) + "\n").encode(),
        "CITATION.cff": citation_text.encode(),
        FACTS: (json.dumps(facts, indent=2, sort_keys=True) + "\n").encode(),
        "calibration/calibrate.json": (root / "release-receipts/calibration.json").read_bytes(),
    }
    for name in ("LICENSE", "DATASET_LICENSE.md"):
        outputs[name] = (source_root / name).read_bytes()
    for relative, content in outputs.items():
        write_staged_bytes(root / relative, content)
    receipt = {
        "schema_version": "passage-successor-publication-v1",
        "dataset_version": VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "status": "prepared",
        "scope": facts["scope"],
        "input_sha256": inputs,
        "files_sha256": {relative: _hash(root / relative) for relative in outputs},
        "facts_sha256": _hash(root / FACTS),
        "learned_results": "not_run",
        "upload": "not_run",
    }
    write_staged_bytes(root / RECEIPT, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return verify_successor_publication_metadata(root)


def verify_successor_publication_metadata(root: str | Path) -> dict:
    """Recheck metadata bytes and their scientific source identities before readiness."""
    root = Path(root).resolve()
    receipt = _json(root / RECEIPT)
    if receipt.get("schema_version") != "passage-successor-publication-v1" or receipt.get("dataset_version") != VERSION:
        raise ValueError("Release publication metadata receipt version mismatch")
    for relative, expected in {**receipt["input_sha256"], **receipt["files_sha256"]}.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Release publication metadata hash mismatch: {relative}")
    facts, inputs = _facts(root)
    if inputs != receipt["input_sha256"] or _json(root / FACTS) != facts:
        raise ValueError("Release publication scientific facts differ from verified receipts")
    from passage.export import _validate_croissant, _validate_readme_card

    for valid, message in (
        _validate_croissant(root / "croissant.json"),
        _validate_readme_card(root / "README.md", set(facts["configurations"])),
    ):
        if not valid:
            raise ValueError(f"Invalid release publication metadata: {message}")
    croissant = _json(root / "croissant.json")
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    if croissant.get("version") != PUBLIC_VERSION or citation.get("version") != PUBLIC_VERSION:
        raise ValueError("Release publication version differs across metadata files")
    expected_range = f"{facts['calibration']['min_m']!r} m to {facts['calibration']['max_m']!r} m"
    if croissant.get("description") != _description(facts) or {
        record["name"] for record in croissant["recordSet"]
    } != set(facts["configurations"]):
        raise ValueError("Release publication scientific metadata differs from verified facts")
    for filename in ("README.md", "DATASHEET.md"):
        content = (root / filename).read_text()
        if (
            expected_range not in content
            or _description(facts) not in content
            or json.dumps(facts["graphs"], indent=2, sort_keys=True) not in content
            or json.dumps(facts["geographic_policy"], indent=2, sort_keys=True) not in content
        ):
            raise ValueError("Release publication scientific metadata differs from verified facts")
    return {**receipt, "status": "verified", "receipt_sha256": _hash(root / RECEIPT)}
