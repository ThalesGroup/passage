"""Write a two-task analytic Parquet fixture for the external CLI example."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from passage.utils import encode_tensor_blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/eval-example/data')
    args = parser.parse_args()
    directory = Path(args.output) / '3x3' / 'validation'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'data-00000.parquet'
    if target.exists():
        raise FileExistsError(target)
    rows = []
    for index in range(2):
        tensor = np.zeros((3, 3, 3), dtype=np.float32)
        tensor[:, :, 1] = -1
        tensor[0, 0, 1] = 0
        tensor[2, 2, 1] = 1
        rows.append({'id': f'analytic-example-{index}', 'tensor': encode_tensor_blob(tensor),
                     'path_astar_elevation_obstacles': [[0, 0], [1, 1], [2, 2]],
                     'metadata': {'resolution': 3, 'split': 'validation', 'index': index,
                                  'cluster_id': f'analytic-block-{index}',
                                  'calibration': {'min': 0., 'max': 1.},
                                  'markers': {'start_point': {'i': 0, 'j': 0}, 'end_point': {'i': 2, 'j': 2}},
                                  'solvers': {'astar_elevation': {'cost_model': 'elevation', 'weight': 62.4,
                                                                'diagonal_movement': 'always'}},
                                  'paths': {'astar_elevation': {'obstacles': {'cost': float(60 * np.sqrt(2))}}}}})
    pq.write_table(pa.Table.from_pylist(rows), target)
    print(target.resolve())


if __name__ == '__main__':
    main()
