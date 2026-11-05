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

"""Host-only, admitted S input reader. No label or roster fingerprint crosses to the target."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from passage.dataset import decode_scene_inputs
from passage_eval.io import roster as roster_io
from passage_eval.protocol.admission import RULE8_NATIVE_PROFILE, grant_test_admission
from passage_eval.protocol.guards import refuse_test_roster

_INPUT_COLUMNS = [
    "id",
    "tensor",
    "metadata.resolution",
    "metadata.split",
    "metadata.index",
    "metadata.calibration",
    "metadata.markers",
    "metadata.tile",
    "metadata.solvers",
]


def admitted_requests(record_path, evidence_dir, roster_path, dataset_root, budget, *, rule8_scope=None):
    """Yield raw requests in the admitted S order, reading only ordinary columns.

    Admission is obtained before the roster or dataset are opened. This iterator
    is host-only; transport must serialize only raw arrays, coordinates and IDs.
    A failed or changed grant stops the stream, including between requests.
    """
    grant = grant_test_admission(record_path, evidence_dir)
    if grant.report.admitted.get("profile") == RULE8_NATIVE_PROFILE:
        from passage_eval.campaign.direct.rule8_scope import Rule8Scope

        if not isinstance(rule8_scope, Rule8Scope):
            raise ValueError("Rule8 requires the pinned global plan before opening the roster")
        rule8_scope.check_record(grant)
    elif rule8_scope is not None:
        raise ValueError("global Rule8 plan cannot be applied to a descriptive grant")
    roster = roster_io.load_roster(roster_path, expected_sha256=grant.admissible_roster_sha256)
    refuse_test_roster(
        roster, allow_test=True, admission=grant, require_admitted_roster=True, context="the direct native input reader"
    )
    if roster["dataset"]["split"] != "test" or len(roster["tasks"]) != 1024:
        raise ValueError("direct campaign admits exactly the sealed 1024-scene test roster S")
    grant.ensure_unchanged()
    root = Path(dataset_root)
    for task in roster["tasks"]:
        current = root
        for part in Path(task["source_file"]).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("roster source path contains a symlink")
    roster_io._check_root(roster, root, require_receipt=True, verify_hashes=True)
    import pyarrow.parquet as pq

    files = {record["path"]: record for record in roster["source_files"]}
    handles = {}
    try:
        for task in roster["tasks"]:
            grant.ensure_unchanged()
            name = task["source_file"]
            if name not in handles:
                parquet = pq.ParquetFile(root / name)
                handles[name] = parquet
                if parquet.metadata.num_rows != files[name]["n_rows"]:
                    raise ValueError("Parquet row count differs from roster")
            parquet = handles[name]
            offset = task["row_index"]
            for group in range(parquet.metadata.num_row_groups):
                count = parquet.metadata.row_group(group).num_rows
                if offset < count:
                    break
                offset -= count
            else:
                raise ValueError("roster row is outside its shard")
            row = parquet.read_row_group(group, columns=_INPUT_COLUMNS).slice(offset, 1).to_pylist()[0]
            if row.get("id") != task["sample_id"]:
                raise ValueError("roster sample ID differs")
            decoded = decode_scene_inputs(row, resolution=roster["dataset"]["resolution"], split="test")
            cell = roster["cell"]
            request = roster_io.inputs_request(
                decoded,
                cost_model=cell["cost_model"],
                cost_weight=cell["cost_weight"],
                task_variant=cell["task_variant"],
            )
            if roster_io.request_input_identity(request) != task["input_identity"]:
                raise ValueError("ordinary input identity differs from roster")
            yield replace(request, metadata={}, input_tensor=None, budget=budget)
    finally:
        for parquet in handles.values():
            parquet.close()
