#!/usr/bin/env python3
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

"""Fail-closed edge runtime checks; device probing is opt-in.

This reports local compatibility evidence only. It does not establish target
identity, accelerator placement, numerical parity or safety of untrusted code.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).with_name('profiles.json')


def load_profiles() -> dict[str, dict[str, Any]]:
    data = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    if data.get('schema') != 'passage-edge-runtime-profiles/v1':
        raise ValueError('unsupported runtime profile schema')
    profiles = data.get('profiles')
    expected = {'thor': ('tensorrt_thor_gpu', 'aarch64'),
                'orin': ('tensorrt_orin_gpu', 'aarch64'),
                'tiger': ('openvino_tiger_gpu', 'x86_64')}
    packages = {'numpy': '2.5.4', 'numba': '0.68.0', 'llvmlite': '0.50.0'}
    nvidia_sdk = {'tensorrt': {'exact': '10.16.2.10'}, 'cuda-python': {'exact': '13.2.0'},
                  'cuda-bindings': {'exact': '13.2.0'}}
    if not isinstance(profiles, dict) or set(profiles) != set(expected):
        raise ValueError('unexpected runtime profile set')
    for name, (backend, arch) in expected.items():
        profile = profiles[name]
        if not isinstance(profile, dict) or (profile.get('backend'), profile.get('architecture')) != (backend, arch):
            raise ValueError('profile backend or architecture mismatch')
        expected_sdk = nvidia_sdk if name in {'thor', 'orin'} else {'openvino': {'exact': '2026.4.1'}}
        if profile.get('python') != [3, 12, 3] or profile.get('packages') != packages or profile.get('sdk') != expected_sdk:
            raise ValueError('profile version requirements mismatch')
    return profiles


def has_forbidden_loaded(names: Any) -> bool:
    return any(name == 'onnxruntime' or name.startswith('onnxruntime.') for name in names)


def _onnxruntime_present() -> bool:
    """Check imports and installed distributions without importing the SDK."""
    if has_forbidden_loaded(sys.modules):
        return True
    try:
        if importlib.util.find_spec('onnxruntime') is not None:
            return True
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get('Name', '')
            if re.sub(r'[-_.]+', '-', name).lower().startswith('onnxruntime'):
                return True
    except (ImportError, ValueError, OSError):
        # A broken import/distribution inventory is not evidence of absence.
        return True
    return False


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 - a broken native import must fail closed
        # Do not print paths or environment-dependent exception contents.
        return False


def observe(profile: str) -> dict[str, Any]:
    spec = load_profiles()[profile]
    names = [*spec['packages'], *spec['sdk']]
    versions = {name: _version(name) for name in names}
    return {'machine': platform.machine(), 'python_version': tuple(sys.version_info[:3]),
            'versions': versions, 'numpy_import': _can_import('numpy'),
            'numba_import': _can_import('numba'),
            'onnxruntime_present': _onnxruntime_present()}


def evaluate(profile: str, backend: str, machine: str, python_version: tuple[int, int, int],
             versions: dict[str, str | None], numpy_import: bool, numba_import: bool,
             onnxruntime_present: bool) -> dict[str, Any]:
    spec = load_profiles()[profile]
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, actual: Any = None, expected: Any = None) -> None:
        checks.append({'name': name, 'ok': bool(ok), 'actual': actual, 'expected': expected})

    normalized_machine = {'arm64': 'aarch64', 'amd64': 'x86_64'}.get(machine.lower(), machine.lower())
    check('backend', backend == spec['backend'], backend, spec['backend'])
    check('architecture', normalized_machine == spec['architecture'], normalized_machine, spec['architecture'])
    expected_python = tuple(spec['python'])
    check('python', tuple(python_version) == expected_python, list(python_version), list(expected_python))
    for library, imported in (('numpy', numpy_import), ('numba', numba_import)):
        check(f'{library}_installed', bool(versions.get(library)), versions.get(library), 'installed')
        check(f'{library}_import', imported, bool(imported), True)
    for library, expected_version in spec['packages'].items():
        check(f'version:{library}', versions.get(library) == expected_version,
              versions.get(library), expected_version)
    check('onnxruntime_absent', not onnxruntime_present, bool(onnxruntime_present), False)
    for name, requirement in spec['sdk'].items():
        actual = versions.get(name)
        if 'exact' in requirement:
            expected = requirement['exact']
            matched = actual == expected
        else:
            prefix = requirement['prefix']
            expected = prefix + '*'
            # Require a numeric patch component, not arbitrary prefix strings.
            matched = bool(actual and re.fullmatch(re.escape(prefix) + r'\d+', actual))
        check(f'sdk:{name}', matched, actual, expected)
    return {'schema': 'passage-edge-runtime-guard/v1', 'profile': profile,
            'backend': backend, 'status': 'ok' if all(c['ok'] for c in checks) else 'failed',
            'checks': checks, 'gpu_probe': 'not_run',
            'scope': 'metadata and CPU imports only; model execution and target identity not verified'}


def probe_gpu(profile: str) -> bool:
    """Explicitly initialize the requested SDK and enumerate its GPU devices."""
    try:
        if profile in {'thor', 'orin'}:
            runtime = importlib.import_module('cuda.bindings.runtime')
            status, count = runtime.cudaGetDeviceCount()
            return int(status) == 0 and int(count) > 0
        if profile == 'tiger':
            ov = importlib.import_module('openvino')
            # A single-GPU OpenVINO installation may enumerate its GPU as
            # either GPU or GPU.0. GPU.1 and composite devices are not this
            # profile's declared target.
            return any(str(device) in {'GPU', 'GPU.0'} for device in ov.Core().available_devices)
    except Exception:  # noqa: BLE001 - vendor runtime exceptions vary by SDK
        return False
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=('thor', 'orin', 'tiger'))
    parser.add_argument('--backend', help='must be the native GPU backend for the profile')
    parser.add_argument('--gpu-smoke', action='store_true', help='initialize SDK and enumerate GPU explicitly')
    parser.add_argument('--json', action='store_true', help='output JSON (the default)')
    args = parser.parse_args(argv)
    try:
        spec = load_profiles()[args.profile]
        o = observe(args.profile)
        result = evaluate(args.profile, args.backend or spec['backend'], o['machine'], o['python_version'],
                          o['versions'], o['numpy_import'], o['numba_import'], o['onnxruntime_present'])
        if args.gpu_smoke and result['status'] == 'ok':
            detected = probe_gpu(args.profile)
            result['gpu_probe'] = 'detected' if detected else 'failed'
            result['checks'].append({'name': 'gpu_enumeration', 'ok': detected,
                                     'actual': detected, 'expected': True})
            if not detected:
                result['status'] = 'failed'
            result['scope'] = 'SDK device enumeration only; model execution and target identity not verified'
        elif args.gpu_smoke:
            result['gpu_probe'] = 'not_run_preflight_failed'
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        result = {'schema': 'passage-edge-runtime-guard/v1', 'profile': args.profile,
                  'status': 'failed', 'error': 'profile or runtime inventory unavailable'}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
