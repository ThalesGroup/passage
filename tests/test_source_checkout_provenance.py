"""Release provenance must be complete before any generation or export work."""

import hashlib
import shutil
from types import SimpleNamespace

import pytest

import passage.export as export
import passage.regenerate as regenerate


def source_checkout(tmp_path):
    root = tmp_path / "checkout"
    contents = {
        "src/passage/__init__.py": b"version = 1\n",
        "src/passage/regenerate.py": b"producer = 1\n",
        "src/passage-eval/__init__.py": b"metric = 1\n",
        "notebooks/example.ipynb": b'{"cells": []}\n',
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "uv.lock": b"version = 1\n",
    }
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root, {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}


@pytest.mark.parametrize("alias", [False, True])
def test_complete_checkout_keeps_full_source_and_producer_rosters(tmp_path, monkeypatch, alias):
    root, expected = source_checkout(tmp_path)
    package = root
    if alias:
        package = tmp_path / "checkout-alias"
        package.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(regenerate, "__file__", str(package / "src/passage/regenerate.py"))
    assert regenerate.code_identity() == expected
    assert regenerate._producer_hashes() == {
        "passage/__init__.py": expected["src/passage/__init__.py"],
        "passage/regenerate.py": expected["src/passage/regenerate.py"],
        "passage-eval/__init__.py": expected["src/passage-eval/__init__.py"],
        "pyproject.toml": expected["pyproject.toml"],
        "uv.lock": expected["uv.lock"],
    }


@pytest.mark.parametrize(
    "missing", ["src/passage", "src/passage-eval", "src/passage-eval/__init__.py", "pyproject.toml", "uv.lock"]
)
@pytest.mark.parametrize("helper", ["code_identity", "_producer_hashes"])
def test_incomplete_checkout_rejects_partial_provenance(tmp_path, monkeypatch, missing, helper):
    root, _ = source_checkout(tmp_path)
    target = root / missing
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    monkeypatch.setattr(regenerate, "__file__", str(root / "src/passage/regenerate.py"))
    with pytest.raises(RuntimeError, match="source checkout"):
        getattr(regenerate, helper)()


def test_regenerate_rejects_installed_layout_before_reading_inputs_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(regenerate, "__file__", str(tmp_path / "site-packages/passage/regenerate.py"))
    before = set(tmp_path.rglob("*"))
    with pytest.raises(RuntimeError, match="source checkout"):
        regenerate.run_regenerate(
            config_filepath=tmp_path / "missing-config.yaml",
            dataset_dirpath=tmp_path / "release",
            scratch_dirpath=tmp_path / "scratch",
            tiles_dirpath=tmp_path / "tiles",
            calibration_filepath=tmp_path / "calibration.json",
        )
    assert set(tmp_path.rglob("*")) == before


def test_successor_export_rejects_installed_layout_before_reading_inputs_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "__file__", str(tmp_path / "site-packages/passage/export.py"))
    context = SimpleNamespace(
        chunk_sizes={64: 1},
        dataset_path=tmp_path / "raw",
        export_dirpath=tmp_path / "release",
        config_file=tmp_path / "missing-config.yaml",
        calibration_file=tmp_path / "calibration.json",
    )
    before = set(tmp_path.rglob("*"))
    with pytest.raises(RuntimeError, match="source checkout"):
        export.export_successor_batch(context, 64, [0])
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("consumer", ["code_identity", "_producer_hashes", "run_regenerate", "export_successor_batch"])
def test_wheel_inside_checkout_cannot_claim_the_checkout_sources(tmp_path, monkeypatch, consumer):
    root, _ = source_checkout(tmp_path)
    vendor = root / "vendor/passage"
    vendor.mkdir(parents=True)
    for name, module in (("regenerate", regenerate), ("export", export)):
        path = vendor / f"{name}.py"
        path.write_text("installed_producer = True\n")
        monkeypatch.setattr(module, "__file__", str(path))
    before = set(tmp_path.rglob("*"))
    with pytest.raises(RuntimeError, match="source checkout"):
        if consumer == "export_successor_batch":
            export.export_successor_batch(SimpleNamespace(), 64, [0])
        elif consumer == "run_regenerate":
            regenerate.run_regenerate(config_filepath=tmp_path / "missing-config.yaml")
        else:
            getattr(regenerate, consumer)()
    assert set(tmp_path.rglob("*")) == before
