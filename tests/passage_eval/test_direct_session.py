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

import json
import numpy as np
import pytest
from passage_eval.protocol.contracts import PlanRequest, BudgetSpec
from passage_eval.campaign.direct.direct_session import run_direct_session


class Runtime:
    def __init__(self):
        self.calls = 0
        self.syncs = 0

    def predict(self, tensor):
        self.calls += 1
        return np.zeros((1, 1, 3, 3), np.float32)

    def synchronize(self):
        self.syncs += 1


class Decoders:
    def build_input(self, terrain, obstacles, sr, sc, gr, gc, low, high):
        return np.full((1, 3, 3, 3), terrain[0, 0], np.float32)

    def reg_bellman_descent(self, field, obstacles, start, goal):
        code = int(self.last_code)
        return {
            "status": "no_path" if code == 18 else "ok",
            "path": [] if code == 18 else [start, (1, 1), goal],
            "fallback": code == 19,
            "descent_steps": 2,
            "fallback_expansions": 1 if code == 19 else 0,
        }

    @property
    def last_code(self):
        return self._code


class RecordingRuntime(Runtime):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        self.seen = []

    def predict(self, tensor):
        self.dec._code = tensor[0, 0, 0, 0]
        self.seen.append(float(self.dec._code))
        return super().predict(tensor)


def req(i):
    return PlanRequest(
        str(i),
        np.full((3, 3), float(i)),
        np.zeros((3, 3), bool),
        (0, 0),
        (2, 2),
        "elevation",
        62.4,
        budget=BudgetSpec(wall_time_ms=5),
    )


def task(request):
    from passage_eval.campaign.direct.raw_identity import raw_digest

    return {"sample_id": request.sample_id, "raw_input_sha256": raw_digest(request)}


def test_single_runtime_spans_blocks_and_retains_failures(tmp_path):
    dec = Decoders()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = [req(i) for i in range(16, 20)]
    tasks = [task(x) for x in measured]
    # Each invocation takes 2ms, except sample 17, which returns after 8ms.
    ticks = iter(
        [x for _ in range(16) for x in (0, 1_000_000)] + [0, 2_000_000, 0, 8_000_000, 0, 2_000_000, 0, 2_000_000]
    )
    result = run_direct_session(
        rt,
        dec,
        "reg",
        warm,
        [measured[:2], measured[2:]],
        tasks,
        tmp_path / "journal.jsonl",
        clock_ns=lambda: next(ticks),
    )
    assert rt.calls == rt.syncs == 20
    assert rt.seen[:16] == [0.0] * 16 and rt.seen[16:] == [16.0, 17.0, 18.0, 19.0]
    assert result["status"] == "complete" and result["warmups_completed"] == 16
    rows = [
        json.loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "request"
    ]
    assert [row["status"] for row in rows] == ["ok", "timeout", "no_path", "ok"]
    assert [row["elapsed_ns"] for row in rows] == [2_000_000, 8_000_000, 2_000_000, 2_000_000]
    assert rows[-1]["fallback"] is True


def test_incomplete_input_preserves_journal_and_fails_closed(tmp_path):
    dec = Decoders()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = [req(16), req(17)]
    tasks = [task(x) for x in measured]
    with pytest.raises(ValueError, match="incomplete"):
        run_direct_session(rt, dec, "reg", warm, [measured[:1]], tasks, tmp_path / "journal.jsonl")
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) >= 2


def test_interruption_leaves_ambiguous_started_marker(tmp_path):
    class InterruptingRuntime(RecordingRuntime):
        def predict(self, tensor):
            if self.calls == 16:
                raise KeyboardInterrupt()
            return super().predict(tensor)

    dec = Decoders()
    rt = InterruptingRuntime(dec)
    warm = [req(0)]
    measured = [req(16)]
    tasks = [task(x) for x in measured]
    with pytest.raises(KeyboardInterrupt):
        run_direct_session(rt, dec, "reg", warm, [measured], tasks, tmp_path / "journal.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert rows[-1] == {"kind": "request_started", "sample_id": "16", "sequence": 0}


def test_failed_sync_is_ambiguous_not_candidate_failure(tmp_path):
    class BrokenSync(RecordingRuntime):
        def synchronize(self):
            self.syncs += 1
            if self.syncs == 17:
                raise RuntimeError("GPU synchronization failed")

    dec = Decoders()
    rt = BrokenSync(dec)
    warm = [req(0)]
    measured = [req(16)]
    tasks = [task(x) for x in measured]
    with pytest.raises(RuntimeError, match="drain"):
        run_direct_session(rt, dec, "reg", warm, [measured], tasks, tmp_path / "journal.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "request_started"


def test_checkpoint_reservations_bound_ambiguity(tmp_path):
    dec = Decoders()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = [req(i) for i in range(16, 20)]
    tasks = [task(x) for x in measured]
    run_direct_session(rt, dec, "reg", warm, [measured], tasks, tmp_path / "journal.jsonl", checkpoint_size=2)
    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [(x["start"], x["end"]) for x in events if x["kind"] == "reservation"] == [(0, 2), (2, 4)]
    assert [x["n_completed"] for x in events if x["kind"] == "checkpoint"] == [2, 4]


def test_interrupted_block_marks_uncommitted_range_ambiguous(tmp_path):
    from passage_eval.campaign.direct.direct_session import inspect_journal

    class InterruptingRuntime(RecordingRuntime):
        def predict(self, tensor):
            if self.calls == 19:
                raise KeyboardInterrupt()
            return super().predict(tensor)

    dec = Decoders()
    rt = InterruptingRuntime(dec)
    warm = [req(0)]
    measured = [req(i) for i in range(16, 21)]
    tasks = [task(x) for x in measured]
    with pytest.raises(KeyboardInterrupt):
        run_direct_session(rt, dec, "reg", warm, [measured], tasks, tmp_path / "journal.jsonl", checkpoint_size=2)
    audit = inspect_journal(tmp_path / "journal.jsonl")
    assert audit["committed_prefix"] == 2
    assert audit["ambiguous_range"] == [2, 4]
    assert audit["visible_uncommitted"] == [2]
    assert audit["automatic_resume_allowed"] is False


def test_each_candidate_gets_new_runtime_and_releases_it(tmp_path):
    from passage_eval.campaign.direct.direct_session import execute_direct_candidate

    made = []

    class Closable(RecordingRuntime):
        def __init__(self, dec):
            super().__init__(dec)
            self.closed = False

        def close(self):
            self.closed = True

    for n in range(2):
        dec = Decoders()

        def factory():
            rt = Closable(dec)
            made.append(rt)
            return rt

        warm = [req(0)]
        measured = [req(16)]
        result = execute_direct_candidate(
            factory, dec, "reg", warm, [measured], [task(measured[0])], tmp_path / f"journal-{n}.jsonl"
        )
        assert result["released"] is True
    assert len(made) == 2 and made[0] is not made[1]
    assert all(rt.closed and rt.calls == 17 and rt.syncs == 18 for rt in made)


def test_prebuilt_tensor_or_metadata_refused(tmp_path):
    from dataclasses import replace

    dec = Decoders()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = replace(req(16), metadata={"task_variant": "obstacles"})
    with pytest.raises(ValueError, match="raw arrays only"):
        run_direct_session(rt, dec, "reg", warm, [[measured]], [task(measured)], tmp_path / "journal.jsonl")


def test_fractional_raw_path_is_not_silently_rounded(tmp_path):
    class Fractional(Decoders):
        def reg_bellman_descent(self, field, obstacles, start, goal):
            return {"status": "ok", "path": [start, (1.5, 1), goal], "fallback": False}

    dec = Fractional()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = [req(16)]
    run_direct_session(rt, dec, "reg", warm, [measured], [task(measured[0])], tmp_path / "journal.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [r for r in rows if r["kind"] == "request"][0]["path"][1] == [1.5, 1.0]


def test_unsupported_work_budget_is_refused(tmp_path):
    from dataclasses import replace

    dec = Decoders()
    rt = RecordingRuntime(dec)
    warm = [req(0)]
    measured = replace(req(16), budget=BudgetSpec(wall_time_ms=5, max_queue_operations=1))
    with pytest.raises(ValueError, match="unsupported"):
        run_direct_session(rt, dec, "reg", warm, [[measured]], [task(measured)], tmp_path / "journal.jsonl")


def test_killed_blocking_call_leaves_explicit_ambiguous_reservation(tmp_path):
    import multiprocessing
    import time
    from passage_eval.campaign.direct.direct_session import inspect_journal

    journal = tmp_path / "journal.jsonl"

    def simulate_blocked_session():
        class Blocking(RecordingRuntime):
            def predict(self, tensor):
                if self.calls == 16:
                    time.sleep(30)
                return super().predict(tensor)

        dec = Decoders()
        runtime = Blocking(dec)
        warm = [req(0)]
        measured = [req(16), req(17)]
        run_direct_session(
            runtime, dec, "reg", warm, [measured], [task(x) for x in measured], journal, checkpoint_size=2
        )

    child = multiprocessing.Process(target=simulate_blocked_session)
    child.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if journal.exists() and '"kind":"request_started"' in journal.read_text():
            break
        time.sleep(0.01)
    else:
        child.terminate()
        child.join()
        pytest.fail("synthetic child did not start its measured request")
    child.terminate()
    child.join(5)
    assert not child.is_alive()
    state = inspect_journal(journal)
    assert state["committed_prefix"] == 0 and state["ambiguous_range"] == [0, 2]
    assert state["automatic_resume_allowed"] is False and state["complete"] is False


def test_session_markers_bind_process_and_stay_outside_request_clock(tmp_path):
    import os
    import re

    dec = Decoders()
    runtime = RecordingRuntime(dec)
    measured = req(16)
    ticks = iter([x for _ in range(17) for x in (0, 1_000_000)])
    run_direct_session(
        runtime,
        dec,
        "reg",
        [req(0)],
        [[measured]],
        [task(measured)],
        tmp_path / "journal.jsonl",
        clock_ns=lambda: next(ticks),
    )
    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    start = events[0]
    end = next(event for event in events if event["kind"] == "session_complete")
    assert re.fullmatch(r"[0-9a-f]{32}", start["session_id"])
    assert start["session_id"] == end["session_id"]
    assert start["namespace_pid"] == end["namespace_pid"] == os.getpid()
    assert 0 < start["marker_monotonic_ns"] <= end["marker_monotonic_ns"]
    assert start["marker_unix_ns"] > 0 and end["marker_unix_ns"] > 0
    assert [event["elapsed_ns"] for event in events if event["kind"] == "request"] == [1_000_000]


@pytest.mark.parametrize("release_fails", [False, True])
def test_existing_journal_remains_unchanged_when_exclusive_open_fails(tmp_path, release_fails):
    from passage_eval.campaign.direct.direct_session import execute_direct_candidate

    class Closable(RecordingRuntime):
        def close(self):
            self.closed = True
            if release_fails:
                raise RuntimeError("synthetic release failure")

    dec = Decoders()
    runtime = Closable(dec)
    journal = tmp_path / "existing.jsonl"
    original = b'{"kind":"session_start","owner":"another session"}\n'
    journal.write_bytes(original)
    measured = req(16)
    expected_error = RuntimeError if release_fails else FileExistsError
    with pytest.raises(expected_error):
        execute_direct_candidate(lambda: runtime, dec, "reg", [req(0)], [[measured]], [task(measured)], journal)
    assert runtime.closed
    assert journal.read_bytes() == original


def test_release_event_remains_bound_to_owned_journal_after_path_replacement(tmp_path):
    from passage_eval.campaign.direct.direct_session import execute_direct_candidate

    journal = tmp_path / "journal.jsonl"
    moved = tmp_path / "original.jsonl"
    foreign = b'{"owner":"other"}\n'

    class ReplacingRuntime(RecordingRuntime):
        def predict(self, tensor):
            if self.calls == 16:
                journal.rename(moved)
                journal.write_bytes(foreign)
            return super().predict(tensor)

        def close(self):
            pass

    dec = Decoders()
    measured = req(16)
    runtime = ReplacingRuntime(dec)
    execute_direct_candidate(lambda: runtime, dec, "reg", [req(0)], [[measured]], [task(measured)], journal)
    assert journal.read_bytes() == foreign
    assert json.loads(moved.read_text().splitlines()[-1])["kind"] == "runtime_released"


def test_invalid_request_does_not_create_a_release_only_journal(tmp_path):
    from passage_eval.campaign.direct.direct_session import execute_direct_candidate

    class Closable(RecordingRuntime):
        def close(self):
            self.closed = True

    dec = Decoders()
    runtime = Closable(dec)
    journal = tmp_path / "journal.jsonl"
    with pytest.raises(ValueError, match="procedure"):
        execute_direct_candidate(lambda: runtime, dec, "invalid", [req(0)], [], [task(req(16))], journal)
    assert runtime.closed
    assert not journal.exists()


def test_release_failure_is_recorded_in_owned_journal(tmp_path):
    from passage_eval.campaign.direct.direct_session import execute_direct_candidate

    class BrokenClose(RecordingRuntime):
        def close(self):
            raise RuntimeError("synthetic close failure")

    dec = Decoders()
    runtime = BrokenClose(dec)
    journal = tmp_path / "journal.jsonl"
    measured = req(16)
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        execute_direct_candidate(lambda: runtime, dec, "reg", [req(0)], [[measured]], [task(measured)], journal)
    assert json.loads(journal.read_text().splitlines()[-1])["kind"] == "runtime_release_failed"
