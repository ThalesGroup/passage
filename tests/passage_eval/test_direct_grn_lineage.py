"""Pinned GRN case selection; synthetic objects, no devices or input data."""

from types import SimpleNamespace

import pytest

from passage_eval.campaign.direct.grn_lineage import _CASES, _select_case, INPUT_ARCHIVE, REWRITTEN, RESULTS, SOURCE


@pytest.mark.parametrize("key", sorted(_CASES))
def test_checkpoint_and_target_bind_the_expected_engine(key):
    checkpoint, target = key
    expected = _CASES[key]
    bundle = SimpleNamespace(manifest={"lineage": {"checkpoint_sha256": checkpoint}})
    assert _select_case(bundle, {"engine_sha256": expected["engine"]}, target) == expected
    with pytest.raises(ValueError, match="not a pinned case"):
        _select_case(bundle, {"engine_sha256": "0" * 64}, target)
    other = "thor" if target == "orin" else "orin"
    with pytest.raises(ValueError, match="not a pinned case"):
        _select_case(bundle, {"engine_sha256": expected["engine"]}, other)


def test_original_2026_lineage_values_remain_the_same():
    checkpoint = "eee055e084aaaec2aa4befb9db9654882d98fe629bb85e70d24a7bf5575eb79b"
    for target, (result, engine) in RESULTS.items():
        case = _CASES[(checkpoint, target)]
        assert case == {
            "source": SOURCE,
            "rewritten": REWRITTEN,
            "input_archive": INPUT_ARCHIVE,
            "result_archive": result,
            "engine": engine,
        }


def test_checkpoint_substitution_is_rejected():
    original = next(iter(_CASES.values()))
    bundle = SimpleNamespace(manifest={"lineage": {"checkpoint_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="not a pinned case"):
        _select_case(bundle, {"engine_sha256": original["engine"]}, "orin")
