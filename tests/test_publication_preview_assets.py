"""Presentation output must not start without explicitly available assets."""
import importlib.util
from pathlib import Path
import sys

import pytest


def test_preview_missing_assets_fails_before_creating_output(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('preview_renderer', scripts / 'render_publication_previews.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match='presentation assets'):
        module.render(output, assets_dir=tmp_path / 'missing-assets')
    assert not output.exists()
