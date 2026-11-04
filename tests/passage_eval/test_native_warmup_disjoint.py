"""Native warm-up disjointness uses synthetic input identities only."""

import pytest
from passage_eval.io import roster as roster_io
from passage_eval.protocol import admission as A


def task(sample_id, seed):
    identity = {key: f"{seed:064x}" for key in roster_io.CONTENT_IDENTITY_FIELDS}
    return {"sample_id": sample_id, "input_identity": identity}


def test_fixed_warmup_can_be_reused_with_another_evaluated_roster():
    warmup = {"tasks": [task("warm", 1)]}
    measured = {"tasks": [task("a", 2), task("b", 3)]}
    assert A.native_warmup_disjoint(warmup, measured)


@pytest.mark.parametrize("collision", ["id", "content"])
def test_fixed_warmup_rejects_id_or_input_content_collision(collision):
    warmup = {"tasks": [task("warm", 1)]}
    other = task("warm" if collision == "id" else "other", 1 if collision == "content" else 2)
    assert not A.native_warmup_disjoint(warmup, {"tasks": [other]})
