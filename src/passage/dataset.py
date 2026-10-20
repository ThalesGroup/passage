"""Bounded, projected access to core and preview PASSAGE Parquet populations.

A local source never falls back to the network. Remote access is explicit and
requires a revision; receipts remain unverified until source content is checked.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
from typing import Any

SPLITS = {'val': 'validation', 'validation': 'validation', 'calibration': 'calibration', 'test': 'test', 'train': 'train'}
POPULATION_SPLITS = ('train', 'validation', 'calibration', 'test')
DEFAULT_COLUMNS = ['id', 'tensor', 'metadata'] + [
    f'path_astar_{model}_{variant}'
    for model in ('elevation', 'slope', 'energy') for variant in ('free', 'obstacles')
]


def normalize_cell(resolution: int, split: str) -> tuple[int, str]:
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError('resolution must be a positive integer')
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split!r}')
    return resolution, SPLITS[split]


class DatasetSource:
    """Select a local snapshot or a pinned Hugging Face source explicitly."""

    def __init__(self, root: str | Path | None = None, repo: str = 'thalesgroup/passage',
                 revision: str | None = None):
        if root is None and (not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision)):
            raise ValueError('Remote dataset mode requires a pinned 40-hex commit revision')
        self.root = Path(root) if root is not None else None
        self.repo = repo
        self.revision = revision

    def identity(self) -> dict[str, Any]:
        return {'mode': 'local' if self.root is not None else 'hf',
                'root': str(self.root.resolve()) if self.root is not None else None,
                'repo': self.repo, 'revision': self.revision,
                'content_verified': False}

    @staticmethod
    def batch_size_for(resolution: int, columns: list[str] | None = None, requested: int | None = None) -> int:
        """Cap tensor-bearing batches near 48 MiB, with one row at 4096²."""
        if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0):
            raise ValueError('batch_size must be a positive integer')
        count = requested if requested is not None else (256 if columns is not None and 'tensor' not in columns else 16)
        if columns is None or 'tensor' in columns:
            count = min(count, max(1, (48 * 1024 * 1024) // (resolution * resolution * 3 * 4)))
        return count

    def _shards(self, resolution: int, split: str, samples: bool) -> list[Path]:
        assert self.root is not None
        config = f'{resolution}x{resolution}' + ('_samples' if samples else '')
        directory = self.root / config / split
        shards = sorted(directory.glob('*.parquet'))
        if not shards:
            raise FileNotFoundError(f'No released Parquet shards in {directory}')
        return shards

    def iter_rows(self, resolution: int, split: str, columns: list[str] | None = None,
                  samples: bool = False, batch_size: int | None = None,
                  limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield a deterministic prefix; project before reading local/remote batches.

        Default columns exclude preview images. ``limit=None`` explicitly means
        the complete selected split; callers producing pilots must supply a cap.
        """
        resolution, split = normalize_cell(resolution, split)
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError('limit must be a nonnegative integer')
        if columns is not None and (not columns or len(set(columns)) != len(columns)):
            raise ValueError('columns must be nonempty and unique')
        batch_size = self.batch_size_for(resolution, columns, batch_size)
        if limit == 0:
            return
        if limit is not None:
            batch_size = min(batch_size, limit)
        yielded = 0
        if self.root is not None:
            import pyarrow.parquet as pq
            for shard in self._shards(resolution, split, samples):
                with pq.ParquetFile(shard) as parquet:
                    available = parquet.schema_arrow.names
                    projected = columns if columns is not None else [
                        name for name in available if name in {'id', 'tensor', 'metadata'} or name.startswith('path_')
                    ]
                    missing = set(projected) - set(available)
                    if missing:
                        raise ValueError(f'{shard}: missing columns {sorted(missing)}')
                    for batch in parquet.iter_batches(batch_size=batch_size, columns=projected):
                        for row in batch.to_pylist():
                            yield row
                            yielded += 1
                            if limit is not None and yielded >= limit:
                                return
        else:
            from datasets import load_dataset
            projected = columns if columns is not None else DEFAULT_COLUMNS
            # ParquetConfig passes these to the Arrow scanner. select_columns
            # after loading alone would not bound the underlying tensor batches.
            stream = iter(load_dataset(self.repo, name=f'{resolution}x{resolution}' + ('_samples' if samples else ''),
                                       split=split, revision=self.revision, streaming=True,
                                       columns=projected, batch_size=batch_size))
            try:
                while limit is None or yielded < limit:
                    row = next(stream, None)
                    if row is None:
                        break
                    missing = set(projected) - set(row)
                    if missing:
                        raise ValueError(f'Remote row missing columns {sorted(missing)}')
                    yield {name: row[name] for name in projected}
                    yielded += 1
            finally:
                if hasattr(stream, 'close'):
                    stream.close()

    def split_manifest(self, resolution: int, samples: bool = False) -> dict[str, dict[str, Any]]:
        """Census four split headers without reading rows or hashing file contents.

        Missing splits and malformed/inconsistent schemas are explicit. Expected
        release cardinalities require separately pinned configuration evidence;
        this header census never infers them from preview counts.
        """
        import hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq

        normalize_cell(resolution, 'train')
        result = {}
        if self.root is None:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(self.repo, name=f'{resolution}x{resolution}' + ('_samples' if samples else ''),
                                           revision=self.revision)
            features = builder.info.features
            columns = list(features) if features is not None else None
            for split in POPULATION_SPLITS:
                info = builder.info.splits.get(split) if builder.info.splits else None
                result[split] = {'status': 'declared' if info is not None else 'unknown',
                                 'count': info.num_examples if info is not None else None,
                                 'columns': columns, 'content_verified': False, 'expected_count': None,
                                 'evidence': 'remote_declared_metadata', 'shards': None}
            return result
        for split in POPULATION_SPLITS:
            record = {'status': 'available', 'count': 0, 'content_verified': False, 'expected_count': None,
                      'evidence': 'local_parquet_footers', 'shards': [], 'schema_consistent': True}
            try:
                shards = self._shards(resolution, split, samples)
            except FileNotFoundError as exc:
                result[split] = {**record, 'status': 'missing', 'count': None, 'error': str(exc)}
                continue
            baseline = None
            for shard in shards:
                try:
                    with pq.ParquetFile(shard) as parquet:
                        schema = parquet.schema_arrow
                        missing = sorted({'id', 'tensor', 'metadata'} - set(schema.names))
                        wrong_types = []
                        for name, valid in (('id', pa.types.is_string), ('tensor', pa.types.is_binary),
                                            ('metadata', pa.types.is_struct)):
                            if name in schema.names and not valid(schema.field(name).type):
                                # Large offsets preserve the same decoded semantics.
                                if not (name == 'id' and pa.types.is_large_string(schema.field(name).type)
                                        or name == 'tensor' and pa.types.is_large_binary(schema.field(name).type)):
                                    wrong_types.append(name)
                        if missing or wrong_types:
                            record['status'] = 'invalid_schema'
                        if baseline is None:
                            baseline = schema
                        elif not baseline.equals(schema, check_metadata=False):
                            record['schema_consistent'] = False
                            if record['status'] == 'available':
                                record['status'] = 'inconsistent_schema'
                        count = parquet.metadata.num_rows
                        record['count'] += count
                        record['shards'].append({'path': str(shard), 'count': count, 'size_bytes': shard.stat().st_size,
                            'columns': schema.names, 'path_columns': [c for c in schema.names if c.startswith('path_')],
                            'missing_required_columns': missing, 'invalid_column_types': wrong_types,
                            'schema_id': hashlib.sha256(str(schema).encode()).hexdigest()})
                except (OSError, ValueError, pa.ArrowException) as exc:
                    record['status'] = 'unreadable'
                    record['count'] = None
                    record['error'] = str(exc)
                    break
            result[split] = record
        return result

    def get_row(self, resolution: int, split: str, sample_id: str | None = None,
                columns: list[str] | None = None, samples: bool = False) -> dict[str, Any]:
        """Return the first row or match an ID without decoding preceding tensors.

        ID lookup scans only IDs, then reads the matching bounded prefix. Use a
        stored roster for repeated lookups in large populations.
        """
        if sample_id is None:
            rows = self.iter_rows(resolution, split, columns, samples, limit=1)
            try:
                row = next(rows, None)
            finally:
                rows.close()
            if row is None:
                raise KeyError('Selected split is empty')
            return row
        for index, row in enumerate(self.iter_rows(resolution, split, ['id'], samples)):
            if row['id'] == sample_id:
                rows = self.iter_rows(resolution, split, columns, samples, batch_size=1, limit=index + 1)
                try:
                    for selected in rows:
                        pass
                    return selected
                finally:
                    rows.close()
        raise KeyError(f'Unknown sample ID {sample_id!r}')

    def split_counts(self, resolution: int, samples: bool = False) -> dict[str, int]:
        """Read local Parquet footers or remote declared counts, never tensors."""
        normalize_cell(resolution, 'train')
        if self.root is not None:
            import pyarrow.parquet as pq
            counts = {}
            for split in POPULATION_SPLITS:
                total = 0
                for shard in self._shards(resolution, split, samples):
                    with pq.ParquetFile(shard) as parquet:
                        total += parquet.metadata.num_rows
                counts[split] = total
            return counts
        from datasets import load_dataset_builder
        builder = load_dataset_builder(self.repo, name=f'{resolution}x{resolution}' + ('_samples' if samples else ''),
                                       revision=self.revision)
        if not builder.info.splits:
            raise ValueError('Remote source does not declare split counts')
        return {split: builder.info.splits[split].num_examples for split in POPULATION_SPLITS}


def decode_scene_inputs(row: Any, *, resolution: int, split: str) -> dict[str, Any]:
    """Validate/decode ordinary tensor inputs independently of stored route labels."""
    import numpy as np
    import zstandard
    from passage.utils import decode_tensor_blob

    sample_id = str(row.get('id', '<missing-id>'))
    try:
        resolution, split = normalize_cell(resolution, split)
        if not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError('missing or empty sample ID')
        metadata = dict(row['metadata'])
        if metadata['resolution'] != resolution or SPLITS[metadata['split']] != split:
            raise ValueError('resolution or split differs from the requested cell')
        low, high = (float(metadata['calibration'][name]) for name in ('min', 'max'))
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError('invalid global elevation calibration')
        tensor = decode_tensor_blob(bytes(row['tensor']), (resolution, resolution, 3)).copy()
        if not np.isfinite(tensor).all() or np.any((tensor[:, :, 0] < 0) | (tensor[:, :, 0] > 1)):
            raise ValueError('invalid normalized elevation channel')
        if not np.isin(tensor[:, :, 2], [0, 1]).all():
            raise ValueError('obstacle channel must be binary')
        markers = tensor[:, :, 1]
        if not np.isin(markers, [-1, 0, 1]).all() or np.count_nonzero(markers == 0) != 1 or np.count_nonzero(markers == 1) != 1:
            raise ValueError('marker channel must contain exactly one start and one goal')
        points = metadata['markers']
        start = tuple(points['start_point'][key] for key in ('i', 'j'))
        goal = tuple(points['end_point'][key] for key in ('i', 'j'))
        if start == goal or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                                for x in (*start, *goal)):
            raise ValueError('start and goal must be distinct integer cells')
        for point, value in ((start, 0), (goal, 1)):
            if any(x < 0 or x >= resolution for x in point) or markers[point] != value:
                raise ValueError('marker coordinates disagree with the input tensor')
        return {'sample_id': row['id'], 'elevation': tensor[:, :, 0].astype(np.float64) * (high - low) + low,
                'obstacles': tensor[:, :, 2].astype(bool), 'start': start, 'goal': goal,
                'input_tensor': tensor, 'metadata': metadata}
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, zstandard.ZstdError) as exc:
        raise ValueError(f'PASSAGE sample {sample_id}: {exc}') from exc
