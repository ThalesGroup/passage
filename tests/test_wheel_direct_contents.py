"""The wheel must carry the exact direct-runner modules from the source tree."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_direct_wheel.py'


def verifier():
    spec = importlib.util.spec_from_file_location('verify_direct_wheel', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checks_actual_wheel_members_and_bytes(tmp_path):
    source = tmp_path / 'direct'
    source.mkdir()
    (source / '__init__.py').write_bytes(b'')
    (source / 'direct_target.py').write_bytes(b'expected')
    wheel = tmp_path / 'passage.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('passage_eval/campaign/direct/__init__.py', b'')
        archive.writestr('passage_eval/campaign/direct/direct_target.py', b'expected')
    assert verifier().verify(wheel, source) == 2
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('passage_eval/campaign/direct/__init__.py', b'')
    with pytest.raises(ValueError, match='members'):
        verifier().verify(wheel, source)
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('passage_eval/campaign/direct/__init__.py', b'')
        archive.writestr('passage_eval/campaign/direct/direct_target.py', b'changed')
    with pytest.raises(ValueError, match='content'):
        verifier().verify(wheel, source)


def test_offline_built_wheel_contains_runner_and_refuses_packaging_regression(tmp_path):
    """Build the real project twice; the omitted package must fail the verifier."""
    import shutil
    import subprocess

    root = SCRIPT.parents[1]
    source = tmp_path / 'source'
    source.mkdir()
    shutil.copytree(root / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE.md'):
        shutil.copyfile(root / filename, source / filename)

    def build(directory):
        subprocess.run(
            ['uv', 'build', '--offline', '--no-sources', '--wheel', '--out-dir', str(directory), str(source)],
            check=True, capture_output=True, text=True,
        )
        wheels = list(directory.glob('*.whl'))
        assert len(wheels) == 1
        return wheels[0]

    config = source / 'pyproject.toml'
    corrected = config.read_text()
    declaration = '    "passage_eval.campaign.direct",\n'
    assert corrected.count(declaration) == 1
    config.write_text(corrected.replace(declaration, ''))
    broken = build(tmp_path / 'broken')
    with pytest.raises(ValueError, match='members'):
        verifier().verify(broken, source / 'src/passage-eval/campaign/direct')
    config.write_text(corrected)
    fixed = build(tmp_path / 'fixed')
    assert verifier().verify(fixed, source / 'src/passage-eval/campaign/direct') >= 1
