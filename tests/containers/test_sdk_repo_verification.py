"""Check the SDK package inventory before an offline image build."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'containers' / 'jetson-sdk' / 'verify_repo.py'


def load_module():
    spec = importlib.util.spec_from_file_location('verify_sdk_repo', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'ubuntu').mkdir(parents=True)
    (repo / 'jetson').mkdir()
    data = b'one package'
    (repo / 'ubuntu' / 'package.deb').write_bytes(data)
    doc = {'deb_count': 1, 'packages': [{'file': 'ubuntu/package.deb', 'size': len(data),
                                         'sha256': hashlib.sha256(data).hexdigest()}]}
    lock = tmp_path / 'packages.lock.json'
    raw = json.dumps(doc).encode()
    lock.write_bytes(raw)
    return repo, lock, hashlib.sha256(raw).hexdigest()


def test_valid_package_inventory(tmp_path):
    repo, lock, digest = fixture(tmp_path)
    assert load_module().verify(repo, lock, digest) == 1


def test_substitution_and_extra_package_are_refused(tmp_path):
    repo, lock, digest = fixture(tmp_path)
    (repo / 'ubuntu' / 'package.deb').write_bytes(b'evil')
    with pytest.raises(ValueError):
        load_module().verify(repo, lock, digest)
    (repo / 'ubuntu' / 'package.deb').write_bytes(b'one package')
    (repo / 'jetson' / 'extra.deb').write_bytes(b'extra')
    with pytest.raises(ValueError):
        load_module().verify(repo, lock, digest)


def test_lock_substitution_and_symlink_are_refused(tmp_path):
    repo, lock, digest = fixture(tmp_path)
    with pytest.raises(ValueError):
        load_module().verify(repo, lock, '0' * 64)
    source = tmp_path / 'other'
    source.write_bytes(b'one package')
    (repo / 'ubuntu' / 'package.deb').unlink()
    (repo / 'ubuntu' / 'package.deb').symlink_to(source)
    with pytest.raises(ValueError):
        load_module().verify(repo, lock, digest)
