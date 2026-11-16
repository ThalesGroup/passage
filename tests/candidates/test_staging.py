from .test_package import package_fixture


def test_only_declared_artifacts_enter_execution_package(tmp_path):
    from passage_eval.candidates.staging import stage_candidate_package
    from passage_eval.protocol.candidate_package import load_candidate_package

    source = tmp_path / "source"
    source.mkdir()
    path, _ = package_fixture(source)
    (source / "oracle-secret.json").write_text('{"oracle_cost": 42}')
    staged = stage_candidate_package(load_candidate_package(path), tmp_path / "execution")
    assert staged.digest == load_candidate_package(path).digest
    assert not (staged.root / "oracle-secret.json").exists()
    (source / "planner.py").write_text("changed")
    staged.verify()


def test_staging_refuses_overwriting_an_existing_execution_snapshot(tmp_path):
    import pytest
    from passage_eval.candidates.staging import stage_candidate_package
    from passage_eval.protocol.candidate_package import load_candidate_package

    source = tmp_path / "source"
    source.mkdir()
    path, _ = package_fixture(source)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        stage_candidate_package(load_candidate_package(path), destination)
