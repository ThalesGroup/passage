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

"""Verify the direct-runner Python files in a built PASSAGE wheel."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

PREFIX = 'passage_eval/campaign/direct/'


def verify(wheel: Path, source: Path) -> int:
    """Return the checked file count; refuse missing, extra or altered modules."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError('source directory is not regular')
    files = sorted(source.glob('*.py'))
    if not files or not (source / '__init__.py').is_file() or not (source / 'direct_target.py').is_file():
        raise ValueError('direct runner source is incomplete')
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError('source module is not a regular file')
    expected = {PREFIX + path.name: path.read_bytes() for path in files}
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith(PREFIX) and name.endswith('.py')]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('wheel direct module members differ from source')
        for name, contents in expected.items():
            if archive.read(name) != contents:
                raise ValueError('wheel module content differs: ' + name)
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'src/passage-eval/campaign/direct')
    args = parser.parse_args()
    print(f'verified {verify(args.wheel, args.source)} direct runner modules')


if __name__ == '__main__':
    main()
