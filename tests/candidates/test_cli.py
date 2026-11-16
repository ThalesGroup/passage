import json
from .test_package import package_fixture


def test_validate_and_register_cli_leave_operator_record(tmp_path, capsys):
    from passage_eval.cli import main

    path, _ = package_fixture(tmp_path)
    assert main(["candidate", "validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["identity"] == "test-graph@1"
    catalog = tmp_path / "operators.json"
    assert main(["candidate", "register", str(path), "--catalog", str(catalog)]) == 0
    assert "test-graph@1" in json.loads(catalog.read_text())["candidates"]


def test_freeze_rejection_does_not_leave_an_invalid_release(tmp_path):
    import pytest
    from passage_eval.cli import main

    path, doc = package_fixture(tmp_path)
    doc["schema"] = "wrong"
    descriptor = tmp_path / "source.json"
    descriptor.write_text(json.dumps(doc))
    output = tmp_path / "release.json"
    with pytest.raises(SystemExit):
        main(["candidate", "freeze", str(descriptor), "--out", str(output)])
    assert not output.exists()
