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

"""Publication metadata derives release claims from frozen output receipts."""

import json
import re
from datetime import datetime, timezone

import pytest


def test_publication_metadata_rejects_unverified_or_blocked_release(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata
    from passage.export import ReleaseContext

    receipts = tmp_path / "release-receipts"
    receipts.mkdir()
    (receipts / "release-verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "data_status": "verified",
                "dataset_version": "passage-successor-v1",
                "admission": {"status": "blocked", "unresolved_variants": 6},
                "resolutions": {},
            }
        )
    )
    context = ReleaseContext(
        tmp_path / "config.yaml",
        tmp_path / "raw",
        tmp_path / "calibrate.json",
        tmp_path,
        tmp_path,
        {64: 1},
        4,
        None,
        64,
        ["astar_elevation", "astar_slope", "astar_energy"],
    )
    with pytest.raises(ValueError, match="qualified"):
        write_successor_publication_metadata(context)


def _publication_fixture(
    tmp_path, *, resolution=64, tile_count=1, elevation_range=(-12.25, 345.75), geographic_policy=None
):
    import hashlib
    import zipfile
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    import yaml
    from rasterio.transform import from_origin
    from passage.calibrate import run_calibrate
    from passage.export import _build_release_context, _features
    from passage.successor_generation import semantic_hash

    root = tmp_path / "release"
    receipts = root / "release-receipts"
    receipts.mkdir(parents=True)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    source_tiles = [tiles / f"N{index:03d}E000_DSM.tif" for index in range(tile_count)]
    for tile in source_tiles:
        with rasterio.open(
            tile,
            "w",
            driver="GTiff",
            height=1,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 1, 1),
        ) as raster:
            raster.write(np.array([elevation_range], dtype=np.float32), 1)
    calibration = tmp_path / "calibrate"
    run_calibrate(calibration, tiles, num_cpus=0)
    calibration_file = calibration / "calibrate.json"
    (receipts / "calibration.json").write_bytes(calibration_file.read_bytes())
    solvers = [
        {"name": f"astar_{model.removesuffix('_uphill')}", "cost_model": model, "weight": 62.4}
        for model in ("elevation", "slope_uphill", "energy")
    ]
    config = {
        "resolutions": [resolution],
        "num_samples": {resolution: 4},
        "generation": {"dataset_version": "passage-successor-v1"},
        "solvers": solvers,
        "splits": {"ratios": {name: 0.25 for name in ("train", "calibration", "validation", "test")}, "seed": 42},
        "export": {"chunk_size": {resolution: 2}, "sample": {"num_samples": 1}},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    (receipts / "resolved-config.yaml").write_bytes(config_file.read_bytes())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    producer = b"fixture producer source\n"
    with zipfile.ZipFile(receipts / "producer-source.zip", "w") as archive:
        archive.writestr("src/passage/fixture.py", producer)
    provenance = {
        "dataset_version": "passage-successor-v1",
        "configuration": config,
        "config_sha256": sha(config_file),
        "calibration_sha256": sha(calibration_file),
        "sources": {f"{index},0": {"name": tile.name, "sha256": sha(tile)} for index, tile in enumerate(source_tiles)},
        "source_snapshot": {"passage/fixture.py": hashlib.sha256(producer).hexdigest()},
        "nodata_policy": "reject-source-crop",
        "geographic_policy": geographic_policy
        or {"version": "source-tile-blocks-v1", "block_degrees": 5, "buffer_tiles": 1},
        "terrain_normalization": "nearest-angular-center-v1",
        "sample_seeding": "blake2b(seed,resolution,index,split)",
        "tie_policy": "float64-priority-then-row-major-node-strict-improvement-v1",
        "scene_attempts": 100,
        "endpoint_policy": "one-distinct-pair-per-accepted-crop-no-feasibility-selection",
    }
    provenance["provenance_id"] = semantic_hash(provenance)
    (receipts / "generation-provenance.json").write_text(json.dumps(provenance))
    for suffix in ("", "_samples"):
        for split in config["splits"]["ratios"]:
            directory = root / f"{resolution}x{resolution}{suffix}" / split
            directory.mkdir(parents=True)
            features = _features(bool(suffix), [s["name"] for s in solvers])
            row = {
                "id": "0000000000",
                "tensor": b"toy schema fixture",
                "metadata_json": "{}",
                "scene_receipt": "{}",
                "metadata": {"index": 0, "resolution": resolution, "split": split},
            }
            for solver in solvers:
                if suffix:
                    import io
                    from PIL import Image

                    image = io.BytesIO()
                    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image, format="PNG")
                    row[f"image_{solver['name']}"] = {"bytes": image.getvalue(), "path": None}
                for variant in ("free", "obstacles"):
                    row[f"path_{solver['name']}_{variant}"] = []
            pq.write_table(
                pa.Table.from_pylist([row], schema=features.arrow_schema),
                directory / "data.parquet",
                compression="zstd",
            )
    last = receipts / f"batch-{resolution}-0000000000-0000000003.json"
    last.write_text("{}")
    (receipts / f"state-{resolution}.json").write_text(
        json.dumps(
            {
                "generation_provenance_id": provenance["provenance_id"],
                "config_sha256": sha(config_file),
                "calibration_sha256": sha(calibration_file),
                "last_manifest": last.name,
                "last_manifest_sha256": sha(last),
            }
        )
    )
    for prefix in ("cohort", "supervision"):
        (receipts / f"{prefix}-{resolution}.jsonl").write_text("fixture frozen roster\n")
    counts = {split: 1 for split in config["splits"]["ratios"]}
    summary = {
        "requested_scenes": 4,
        "core_counts": counts,
        "sample_counts": counts,
        "successful_route_variants": 23,
        "proved_infeasible_variants": 1,
        "unresolved_variants": 0,
        "scene_status_counts": {f"{split}:complete": 1 for split in counts},
        "variant_status_counts": {
            "train:ok": 5,
            "train:infeasible": 1,
            **{f"{split}:ok": 6 for split in counts if split != "train"},
        },
        "reference_status_counts": {
            "train:exact": 5,
            "train:no_path": 1,
            **{f"{split}:exact": 6 for split in counts if split != "train"},
        },
        "cohort_sha256": sha(receipts / f"cohort-{resolution}.jsonl"),
        "supervision_sha256": sha(receipts / f"supervision-{resolution}.jsonl"),
    }
    (receipts / "release-verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "data_status": "verified",
                "dataset_version": "passage-successor-v1",
                "scope": "full",
                "resolutions": {str(resolution): summary},
                "admission": {"status": "qualified", "unresolved_variants": 0},
            }
        )
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    return _build_release_context(
        raw, calibration_filepath=calibration_file, config_filepath=config_file, export_dirpath=root
    )


def test_successor_metadata_uses_actual_range_counts_and_separate_no_path_outcomes(tmp_path):
    from passage.successor_publication import (
        write_successor_publication_metadata,
        verify_successor_publication_metadata,
    )

    context = _publication_fixture(tmp_path)
    receipt = write_successor_publication_metadata(context)
    root = context.export_dirpath
    card = (root / "README.md").read_text()
    assert "-12.25 m to 345.75 m" in card
    assert "23 successful exact routes" in card
    assert "1 independently proved no-path variants" in card
    assert "not_run" not in card and "No learned-model result" not in card
    assert "sea level" not in card and "237 variants" not in card
    # The dataset card cites no commit revision (40-hex identifiers).
    assert re.search(r"\b[0-9a-fA-F]{40}\b", card) is None
    assert json.loads((root / "croissant.json").read_text())["version"] == "1.0.0"
    assert receipt["status"] == "verified"
    assert verify_successor_publication_metadata(root)["receipt_sha256"] == receipt["receipt_sha256"]
    (root / "README.md").write_text(card.replace("345.75", "8767.0"))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_successor_publication_metadata(root)
    # Even a refreshed file hash cannot bless a scientifically stale range.
    import hashlib

    receipt_path = root / "release-receipts/publication-metadata.json"
    stored = json.loads(receipt_path.read_text())
    stored["files_sha256"]["README.md"] = hashlib.sha256((root / "README.md").read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="scientific metadata"):
        verify_successor_publication_metadata(root)


def test_successor_metadata_rejects_footer_count_different_from_verified_population(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    path = context.export_dirpath / "release-receipts/release-verification.json"
    receipt = json.loads(path.read_text())
    receipt["resolutions"]["64"]["sample_counts"]["test"] = 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="footer count"):
        write_successor_publication_metadata(context)


def test_description_states_all_exact_routes_when_no_variant_lacks_a_path():
    from passage.successor_publication import _description

    facts = {
        "core_scenes": 4,
        "route_variants": 24,
        "preview_rows": 4,
        "successful_route_variants": 24,
        "proved_infeasible_variants": 0,
    }
    text = _description(facts)
    assert "All 24 variants are exact routes" in text
    assert "disconnect are rejected by the generator" in text and "no-path" not in text
    assert "No learned-model result" not in text


def test_publication_preserves_croissant_schema_rai_and_credits(tmp_path):
    import yaml
    from passage.export import _repo_root
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    before = datetime.now(timezone.utc)
    write_successor_publication_metadata(context)
    after = datetime.now(timezone.utc)
    source = json.loads((_repo_root() / "croissant.json").read_text())
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    assert published["@context"] == source["@context"]
    assert published["creditText"] == source["creditText"]
    assert published["creator"] == source["creator"] and len(published["creator"]) == 3
    assert {key for key in published if key.startswith(("rai:", "prov:"))} == {
        key for key in source if key.startswith(("rai:", "prov:"))
    }
    assert "https://earth.jaxa.jp/en/data/policy/" in published["license"]
    assert published["rai:dataCollectionType"] == source["rai:dataCollectionType"]
    assert "24" in published["rai:annotationsPerItem"]
    assert "-12.25 m to 345.75 m" in str(published["rai:dataPreprocessingProtocol"])
    assert "4,400,000" not in json.dumps(published) and "26,400,000" not in json.dumps(published)
    assert before <= datetime.fromisoformat(published["dateModified"]) <= after
    assert datetime.fromisoformat(published["datePublished"]).date() <= after.date()
    source_records = {record["name"]: record for record in source["recordSet"]}
    assert {record["name"] for record in published["recordSet"]} == {"64x64", "64x64_samples"}
    for record in published["recordSet"]:
        original = source_records[record["name"]]
        assert len(record["field"]) == len(original["field"])
        for field, expected in zip(record["field"], original["field"]):
            for key in ("@id", "name", "dataType", "source", "subField", "isArray", "arrayShape"):
                assert field.get(key) == expected.get(key)
    distribution = {item["@id"]: item for item in published["distribution"]}
    assert set(distribution) == {"hf-repository", "64x64-parquet", "64x64_samples-parquet"}
    assert distribution["hf-repository"]["contentUrl"] == "https://huggingface.co/datasets/thalesgroup/passage"
    assert distribution["64x64-parquet"]["containedIn"] == {"@id": "hf-repository"}
    assert "train 1" in distribution["64x64-parquet"]["description"]
    citation = yaml.safe_load((context.export_dirpath / "CITATION.cff").read_text())
    assert citation["version"] == "1.0.0"
    facts = json.loads((context.export_dirpath / "release-receipts/publication-facts.json").read_text())
    assert facts["dataset_version"] == "passage-successor-v1"


def test_publication_provenance_uses_current_calibrated_inventory(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    receipt = write_successor_publication_metadata(context)
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    provenance = published["prov:wasDerivedFrom"][0]["sc:description"]
    assert receipt["status"] == "verified"
    assert "1 calibrated source tile" in provenance
    assert "calibration/calibrate.json" in provenance
    assert "23,993" not in provenance and "19,051" not in provenance
    assert "v4.1" not in provenance and "v3.1" not in provenance and "v4.0" not in provenance


@pytest.mark.parametrize(
    "privacy_receipt",
    [
        None,
        "not a JSON receipt",
        json.dumps(
            {
                "schema": "passage-publication-redactions-v3",
                "status": "verified",
                "files": [{"path": "README.md", "sha256": "0" * 64}],
            }
        ),
    ],
    ids=["absent", "malformed", "unverified-hashes"],
)
def test_publication_does_not_certify_unverified_privacy_redactions(tmp_path, privacy_receipt):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    if privacy_receipt is not None:
        (context.export_dirpath / "release-receipts/privacy-redactions.json").write_text(privacy_receipt)
    receipt = write_successor_publication_metadata(context)
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    privacy = published["rai:personalSensitiveInformation"]
    assert receipt["status"] == "verified"
    assert "were redacted" not in privacy and "privacy-redactions.json" not in privacy
    assert "No human subjects" in privacy
    assert "not established by this metadata" in privacy


@pytest.mark.parametrize(
    "resolution, tile_count, elevation_range, geographic_policy",
    [
        (64, 1, (-12.25, 345.75), {"version": "source-tile-blocks-v1", "block_degrees": 5, "buffer_tiles": 1}),
        (128, 2, (100.5, 300.25), {"version": "fixture-partition-v2", "block_degrees": 2, "buffer_tiles": 0}),
    ],
)
def test_all_publication_rai_claims_follow_current_population(
    tmp_path, resolution, tile_count, elevation_range, geographic_policy
):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(
        tmp_path,
        resolution=resolution,
        tile_count=tile_count,
        elevation_range=elevation_range,
        geographic_policy=geographic_policy,
    )
    write_successor_publication_metadata(context)
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    responsibility = {key: value for key, value in published.items() if key.startswith(("rai:", "prov:"))}
    descriptions = json.dumps(responsibility)
    assert "South America" not in descriptions
    assert "64 to 4096" not in descriptions
    assert "signed int16" not in descriptions and "3600x3600" not in descriptions
    assert "23,993" not in descriptions and "v4.1" not in descriptions
    assert f"{tile_count} calibrated source tile" in descriptions
    assert f"{elevation_range[0]!r} m to {elevation_range[1]!r} m" in descriptions
    assert f"{resolution}x{resolution}" in str(published["rai:dataUseCases"])
    assert json.dumps(geographic_policy, sort_keys=True) in " ".join(published["rai:dataBiases"])


def test_publication_does_not_inherit_template_rai_or_provenance_claims(tmp_path, monkeypatch):
    from passage.export import _repo_root
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    source_root = _repo_root()
    source = tmp_path / "source"
    source.mkdir()
    for name in ("CITATION.cff", "croissant.json", "LICENSE", "DATASET_LICENSE.md"):
        (source / name).write_bytes((source_root / name).read_bytes())
    croissant = json.loads((source / "croissant.json").read_text())
    marker = "UNBOUND_TEMPLATE_CLAIM"
    responsibility_keys = {key for key in croissant if key.startswith(("rai:", "prov:"))}
    for key in responsibility_keys:
        if key.startswith("prov:"):
            croissant[key] = [{"sc:description": marker, "prov:label": marker}]
        elif isinstance(croissant[key], list):
            croissant[key] = [marker]
        elif isinstance(croissant[key], bool):
            croissant[key] = False
        else:
            croissant[key] = marker
    (source / "croissant.json").write_text(json.dumps(croissant))
    monkeypatch.setattr("passage.export._repo_root", lambda: source)
    write_successor_publication_metadata(context)
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    responsibility = {key: value for key, value in published.items() if key.startswith(("rai:", "prov:"))}
    assert responsibility.keys() == responsibility_keys
    assert marker not in json.dumps(responsibility)
    assert published["rai:hasSyntheticData"] is True


def test_publication_rejects_responsibility_fields_without_an_evidence_binding(tmp_path):
    from passage.export import _repo_root
    from passage.successor_publication import _facts, _publication_croissant

    context = _publication_fixture(tmp_path)
    source = json.loads((_repo_root() / "croissant.json").read_text())
    source["rai:unreviewedClaim"] = "A future template assertion without an evidence binding."
    facts, _ = _facts(context.export_dirpath)
    with pytest.raises(ValueError, match="Unbound responsibility metadata"):
        _publication_croissant(source, facts, context.export_dirpath, "example/passage", "main")


@pytest.mark.parametrize(
    "revision, suffix",
    [
        ("main", ""),
        ("v1.0", "/tree/refs%2Ftags%2Fv1.0"),
        ("refs/heads/release", "/tree/refs%2Fheads%2Frelease"),
        ("refs/tags/refs/heads/release", "/tree/refs%2Ftags%2Frefs%2Fheads%2Frelease"),
    ],
)
def test_publication_croissant_can_target_a_huggingface_ref(tmp_path, revision, suffix):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    write_successor_publication_metadata(context, repo_id="example/passage", revision=revision)
    published = json.loads((context.export_dirpath / "croissant.json").read_text())
    container = next(item for item in published["distribution"] if item["@id"] == "hf-repository")
    assert container["contentUrl"] == "https://huggingface.co/datasets/example/passage" + suffix
    from urllib.parse import quote

    hub_revision = revision.split("/", 2)[2] if revision.startswith("refs/") else revision
    assert (
        "https://huggingface.co/datasets/example/passage/blob/" + quote(hub_revision, safe="") + ("/DATASET_LICENSE.md")
        in published["license"]
    )
    if revision != "main":
        mlcroissant = pytest.importorskip("mlcroissant")
        from mlcroissant._src.operation_graph.operations.download import extract_git_info

        assert extract_git_info(container["contentUrl"]) == (
            "https://huggingface.co/datasets/example/passage",
            revision if revision.startswith("refs/") else f"refs/tags/{revision}",
        )
        assert mlcroissant.Dataset(jsonld=published).metadata


@pytest.mark.parametrize("revision", ["", "../main", "x?token=private", "a" * 40])
def test_publication_rejects_unusable_huggingface_refs(tmp_path, revision):
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        write_successor_publication_metadata(context, revision=revision)


def test_publication_rejects_source_croissant_with_different_columns(tmp_path, monkeypatch):
    from passage.export import _repo_root
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    source_root = _repo_root()
    source = tmp_path / "source"
    source.mkdir()
    for name in ("CITATION.cff", "croissant.json", "LICENSE", "DATASET_LICENSE.md"):
        (source / name).write_bytes((source_root / name).read_bytes())
    croissant = json.loads((source / "croissant.json").read_text())
    croissant["recordSet"][0]["field"].pop()
    (source / "croissant.json").write_text(json.dumps(croissant))
    monkeypatch.setattr("passage.export._repo_root", lambda: source)
    with pytest.raises(ValueError, match="schema"):
        write_successor_publication_metadata(context)


def test_generated_croissant_loads_core_and_preview_records(tmp_path):
    mlcroissant = pytest.importorskip("mlcroissant")
    from PIL import Image
    from passage.successor_publication import write_successor_publication_metadata

    context = _publication_fixture(tmp_path)
    write_successor_publication_metadata(context)
    dataset = mlcroissant.Dataset(
        jsonld=str(context.export_dirpath / "croissant.json"), mapping={"hf-repository": str(context.export_dirpath)}
    )
    assert not dataset.metadata.ctx.issues.errors
    assert not dataset.metadata.ctx.issues.warnings
    core = list(dataset.records("64x64"))
    previews = list(dataset.records("64x64_samples"))
    assert len(core) == len(previews) == 4
    assert all(record["64x64/tensor"] == b"toy schema fixture" for record in core)
    for record in previews:
        image = record["64x64_samples/image_astar_elevation"]
        assert isinstance(image, Image.Image) and image.size == (2, 2)
