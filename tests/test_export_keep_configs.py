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

"""Staged exports retain only compatible declarations when explicitly requested."""

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import passage
from passage.export import ExportSteps, prepare_export


@pytest.fixture
def staged_export(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    calibration = tmp_path / "calibrate.json"
    calibration.write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "export:\n  chunk_size: {64: 10, 128: 10}\n  default: 64x64_samples\nsolvers:\n  - name: astar_elevation\n",
        encoding="utf-8",
    )
    export = tmp_path / "export"
    return {
        "dataset_dirpath": dataset,
        "notebooks_dirpath": notebooks,
        "calibration_filepath": calibration,
        "config_filepath": config,
        "export_dirpath": export,
        "repo_id": "example-org/staged-passage",
        "execute_notebooks": False,
        "steps": ExportSteps(skip_validate_croissant=True),
    }


def _stage(export: Path, resolution: int) -> None:
    for suffix in ("", "_samples"):
        split = export / f"{resolution}x{resolution}{suffix}" / "train"
        split.mkdir(parents=True)
        pq.write_table(pa.table({"id": ["0000000000"]}), split / "data-00000.parquet", compression="zstd")


def _frontmatter(export: Path) -> dict:
    return yaml.safe_load((export / "README.md").read_text(encoding="utf-8").split("---", 2)[1])


def _first_stage_then_remove(options: dict) -> Path:
    export = options["export_dirpath"]
    _stage(export, 64)
    assert prepare_export(**options, resolutions=[64]) is True
    shutil.rmtree(export / "64x64")
    shutil.rmtree(export / "64x64_samples")
    _stage(export, 128)
    return export


def test_retains_full_sample_and_default_after_prior_parquets_are_removed(staged_export):
    export = _first_stage_then_remove(staged_export)

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is True

    card = _frontmatter(export)
    configs = {entry["config_name"]: entry for entry in card["configs"]}
    assert set(configs) == {"64x64", "64x64_samples", "128x128", "128x128_samples"}
    assert {entry["config_name"] for entry in card["dataset_info"]} == set(configs)
    assert [name for name, entry in configs.items() if entry.get("default")] == ["64x64_samples"]
    assert configs["64x64"]["data_files"][0] == {"split": "train", "path": "64x64/train/*"}
    assert not (export / "64x64").exists()
    assert not (export / "64x64_samples").exists()


def test_default_export_still_declares_only_current_local_parquets(staged_export):
    export = _first_stage_then_remove(staged_export)

    assert prepare_export(**staged_export, resolutions=[128]) is True

    configs = {entry["config_name"]: entry for entry in _frontmatter(export)["configs"]}
    assert set(configs) == {"128x128", "128x128_samples"}
    assert configs["128x128_samples"]["default"] is True


def test_opt_in_first_stage_needs_no_existing_card(staged_export):
    export = staged_export["export_dirpath"]
    _stage(export, 64)

    assert prepare_export(**staged_export, resolutions=[64], keep_existing_configs=True) is True
    assert {entry["config_name"] for entry in _frontmatter(export)["configs"]} == {"64x64", "64x64_samples"}


@pytest.mark.parametrize(
    "prior_card",
    [
        "# No frontmatter\n",
        "---\nconfigs: [\n---\n",
        "---\nconfigs: []\n---\n",
        "---\nconfigs: [{config_name: ../64x64, data_files: []}]\n---\n",
        "---\nconfigs: [{config_name: 064x064, data_files: []}]\n---\n",
        "---\nconfigs: [{config_name: 64x64, data_files: [{split: train, path: ../64x64/train/*}]}]\n---\n",
    ],
)
def test_malformed_or_incompatible_card_fails_without_overwriting_it(staged_export, prior_card):
    export = staged_export["export_dirpath"]
    _stage(export, 128)
    readme = export / "README.md"
    readme.write_text(prior_card, encoding="utf-8")

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert readme.read_text(encoding="utf-8") == prior_card
    assert not (export / "croissant.json").exists()


def test_retention_rejects_different_repository_before_rewriting_metadata(staged_export):
    export = _first_stage_then_remove(staged_export)
    readme_before = (export / "README.md").read_bytes()
    croissant_before = (export / "croissant.json").read_bytes()
    staged_export["repo_id"] = "example-org/other-passage"

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert (export / "README.md").read_bytes() == readme_before
    assert (export / "croissant.json").read_bytes() == croissant_before
    assert json.loads(croissant_before)["@id"] == "https://huggingface.co/datasets/example-org/staged-passage"


@pytest.mark.parametrize("path", ["128x128/train/*", "/64x64/train/*", "64x64/train/**/*.parquet"])
def test_retention_rejects_paths_outside_the_generated_config_layout(staged_export, path):
    export = _first_stage_then_remove(staged_export)
    card = _frontmatter(export)
    card["configs"][0]["data_files"][0]["path"] = path
    readme = export / "README.md"
    readme.write_text("---\n" + yaml.safe_dump(card) + "---\n# Existing\n", encoding="utf-8")
    before = readme.read_bytes()

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert readme.read_bytes() == before


@pytest.mark.parametrize("suffix", ["", "_samples"])
def test_prior_declaration_does_not_replace_required_selected_parquets(staged_export, suffix):
    export = _first_stage_then_remove(staged_export)
    shutil.rmtree(export / f"128x128{suffix}")
    readme_before = (export / "README.md").read_bytes()

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert (export / "README.md").read_bytes() == readme_before


def test_retention_rejects_changed_solver_schema_before_rewriting_card(staged_export):
    export = _first_stage_then_remove(staged_export)
    readme_before = (export / "README.md").read_bytes()
    config = staged_export["config_filepath"]
    config.write_text(config.read_text().replace("astar_elevation", "astar_energy"), encoding="utf-8")

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert (export / "README.md").read_bytes() == readme_before


def test_retention_rejects_changed_declared_feature_type(staged_export):
    export = _first_stage_then_remove(staged_export)
    card = _frontmatter(export)
    card["dataset_info"][0]["features"][0]["dtype"] = "float32"
    readme = export / "README.md"
    readme.write_text("---\n" + yaml.safe_dump(card) + "---\n# Existing\n", encoding="utf-8")
    before = readme.read_bytes()

    assert prepare_export(**staged_export, resolutions=[128], keep_existing_configs=True) is False
    assert readme.read_bytes() == before


def test_cli_opt_in_retains_card_configs(staged_export, monkeypatch):
    export = _first_stage_then_remove(staged_export)
    monkeypatch.setattr(passage, "prepare_export", lambda **kwargs: prepare_export(execute_notebooks=False, **kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "passage",
            "export",
            "--keep-existing-configs",
            "--resolution",
            "128",
            "--repo-id",
            staged_export["repo_id"],
            "--config",
            str(staged_export["config_filepath"]),
            "--dataset-dirpath",
            str(staged_export["dataset_dirpath"]),
            "--notebooks-dirpath",
            str(staged_export["notebooks_dirpath"]),
            "--calibration-filepath",
            str(staged_export["calibration_filepath"]),
            "--export-dirpath",
            str(export),
            "--no-validate-croissant",
        ],
    )

    passage.main()

    assert {entry["config_name"] for entry in _frontmatter(export)["configs"]} == {
        "64x64",
        "64x64_samples",
        "128x128",
        "128x128_samples",
    }
