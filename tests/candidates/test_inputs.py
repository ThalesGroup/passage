import numpy as np
from .test_scoring import score_fixture


def test_export_contains_only_inputs_and_preserves_roster(tmp_path):
    from passage_eval.campaign.candidate.inputs import export_input_cohort

    run, index = score_fixture(tmp_path)
    out = tmp_path / "inputs.npz"
    receipt = export_input_cohort(index, out)
    with np.load(out, allow_pickle=False) as d:
        assert set(d.files) == {"sample_ids", "terrain", "obstacles", "starts", "goals"}
        assert d["terrain"].dtype == np.float64 and d["obstacles"].dtype == np.bool_
        assert d["sample_ids"].tolist() == ["a"]
    assert receipt["n_expected"] == 1
