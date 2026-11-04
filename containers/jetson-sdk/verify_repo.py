"""Verify an offline SDK package repository against an independently pinned lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(repo: Path, lock: Path, expected_lock_sha256: str) -> int:
    if not re.fullmatch(r'[0-9a-f]{64}', expected_lock_sha256):
        raise ValueError('invalid lock digest')
    if repo.is_symlink() or not repo.is_dir() or lock.is_symlink() or not lock.is_file():
        raise ValueError('repository or lock is not a regular input')
    if _digest(lock) != expected_lock_sha256:
        raise ValueError('lock digest differs')
    document = json.loads(lock.read_bytes())
    packages = document['packages']
    if not isinstance(packages, list) or document['deb_count'] != len(packages) or not packages:
        raise ValueError('invalid package inventory')
    expected = set()
    for entry in packages:
        name, size, digest = entry['file'], entry['size'], entry['sha256']
        if not isinstance(name, str) or not re.fullmatch(r'(ubuntu|jetson)/[A-Za-z0-9.+_~%:-]+\.deb', name):
            raise ValueError('invalid package path')
        if name in expected or type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid package metadata')
        expected.add(name)
        path = repo / name
        if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size != size or _digest(path) != digest:
            raise ValueError('package differs: ' + name)
    actual = set()
    for directory in ('ubuntu', 'jetson'):
        folder = repo / directory
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('invalid package directory')
        for path in folder.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != '.deb':
                raise ValueError('unexpected repository entry')
            actual.add(directory + '/' + path.name)
    if actual != expected:
        raise ValueError('repository package inventory differs')
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('lock', type=Path)
    parser.add_argument('lock_sha256')
    args = parser.parse_args()
    print(f'verified {verify(args.repo, args.lock, args.lock_sha256)} locked packages')


if __name__ == '__main__':
    main()
