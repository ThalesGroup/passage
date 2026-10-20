"""Authored reports are parameterized real-data consumers with no launch cells."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "notebooks/passage-eval").glob("*.ipynb")) + sorted((ROOT / "paper/notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_source_parameters_precede_imports_and_reports_have_no_launch_cells(source):
    notebook = json.loads(source.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "parameters" in code[0]["metadata"].get("tags", [])
    parameters = ast.parse("".join(code[0]["source"]))
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Call)) for n in ast.walk(parameters))
    names = {n.targets[0].id for n in parameters.body if isinstance(n, ast.Assign)}
    assert {"dataset_path", "dataset_repo", "dataset_revision", "output_dir"} <= names
    assert names & {"resolution", "resolutions", "board_resolution"}
    for cell in code:
        # CPython 3.12 feature_version does not reject every PEP 701 f-string.
        assert 'f"{report["' not in "".join(cell["source"])
        ast.parse("".join(cell["source"]), feature_version=(3, 11))
    text = "\n".join("".join(c["source"]) for c in code)
    for forbidden in ("np.random", "random.", "os.chdir", "%pip", "!pip", "subprocess", ".train(", "bench_cli("):
        assert forbidden not in text
    assert "passage.notebook_reports" in text
    assert all(c.get("execution_count") is None and c.get("outputs", []) == [] for c in code)
