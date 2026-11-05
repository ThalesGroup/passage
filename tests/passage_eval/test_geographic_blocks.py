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

import pytest
from passage_eval.protocol.geographic_blocks import source_window_blocks


def test_tasks_sharing_a_five_degree_block_or_a_crossing_share_a_cluster():
    records = [
        {"sample_id": "a", "source_windows": [{"lat": 0, "lon": 0}]},
        {"sample_id": "b", "source_windows": [{"lat": 4, "lon": 4}, {"lat": 4, "lon": 5}]},
        {"sample_id": "c", "source_windows": [{"lat": 1, "lon": 6}]},
        {"sample_id": "d", "source_windows": [{"lat": 20, "lon": 40}]},
    ]
    result = source_window_blocks(records, block_degrees=5)
    assert result["task_blocks"]["a"] == result["task_blocks"]["b"] == result["task_blocks"]["c"]
    assert result["task_blocks"]["d"] != result["task_blocks"]["a"]
    assert result["crossing_task_count"] == 1
    assert result["number_of_blocks"] == 2


def test_antimeridian_longitudes_are_normalized_to_the_same_source_tile():
    records = [
        {"sample_id": "east", "source_windows": [{"lat": 1, "lon": 180}]},
        {"sample_id": "west", "source_windows": [{"lat": 1, "lon": -180}]},
    ]
    result = source_window_blocks(records, block_degrees=5)
    assert result["task_blocks"]["east"] == result["task_blocks"]["west"]


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"sample_id": "a", "source_windows": []}],
        [{"sample_id": "a", "source_windows": [{"lat": True, "lon": 1}]}],
        [{"sample_id": "a", "source_windows": [{"lat": 90, "lon": 1}]}],
        [
            {"sample_id": "a", "source_windows": [{"lat": 0, "lon": 1}]},
            {"sample_id": "a", "source_windows": [{"lat": 2, "lon": 3}]},
        ],
    ],
)
def test_bad_or_unidentified_source_footprints_are_rejected(records):
    with pytest.raises((ValueError, TypeError)):
        source_window_blocks(records, block_degrees=5)
