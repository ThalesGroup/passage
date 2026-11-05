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

"""Verify an offline context against a separately obtained manifest SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root: Path, expected_manifest_sha256: str) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('context root must be a directory, not a symlink')
    if not re.fullmatch(r'[a-f0-9]{64}', expected_manifest_sha256):
        raise ValueError('expected manifest digest must be a lowercase SHA-256')
    manifest_path = root / 'context-manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('manifest must be a regular file')
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError('manifest SHA-256 mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest.get('files_sha256') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('schema') != 'passage-edge-container-context-v1'
            or not isinstance(entries, dict) or not entries):
        raise ValueError('unsupported or empty context manifest')
    for name, expected in entries.items():
        if not isinstance(name, str) or not name or '\\' in name:
            raise ValueError('invalid manifest path')
        relative = PurePosixPath(name)
        if (relative.is_absolute() or str(relative) != name or '..' in relative.parts
                or name == 'context-manifest.json' or not isinstance(expected, str)
                or not re.fullmatch(r'[a-f0-9]{64}', expected)):
            raise ValueError('invalid manifest path or file digest')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlinks are forbidden in the context')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name == 'context-manifest.json':
                continue
            actual.add(name)
            if name not in entries or digest(path) != entries[name]:
                raise ValueError('undeclared or altered context file')
        elif not path.is_dir():
            raise ValueError('nonregular context item')
    if actual != set(entries):
        raise ValueError('context is missing declared files')
    return {'schema': 'passage-edge-container-context-check/v1', 'status': 'ok',
            'manifest_sha256': expected_manifest_sha256, 'file_count': len(actual),
            'scope': 'file identity only; image, SDK and GPU behavior not verified'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('context', type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.context, args.expected_manifest_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
