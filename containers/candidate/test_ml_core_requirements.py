"""Exercise the image guard with real installed-distribution metadata only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


VERIFIER = Path(__file__).with_name("verify_ml_core_requirements.py")
PREPARE = Path(__file__).with_name("prepare_ml_context.py")
BOOTSTRAP = """
import runpy
import sys
prefix, script, wheel = sys.argv[1:]
sys.path.insert(0, prefix)
sys.argv = [script, wheel]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    assert not {"passage", "passage_eval", "torch", "gymnasium", "stable_baselines3", "sb3_contrib"}.intersection(sys.modules)
"""


class CoreRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.site = self.root / "runtime/lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        self.wheel = self.root / "passage-1.0.0-py3-none-any.whl"

    def distribution(self, name, version, requires=(), requires_python=None):
        info = self.site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        lines = ["Metadata-Version: 2.3", f"Name: {name}", f"Version: {version}"]
        if requires_python is not None:
            lines.append(f"Requires-Python: {requires_python}")
        lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
        contents = "\n".join(lines) + "\n\n"
        (info / "METADATA").write_text(contents)
        if name == "passage":
            with zipfile.ZipFile(self.wheel, "w") as wheel:
                wheel.writestr("passage-1.0.0.dist-info/METADATA", contents)
        return info

    def run_guard(self, script=VERIFIER):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-c", BOOTSTRAP, str(self.site), str(script), str(self.wheel)],
            cwd=self.root, env=environment, text=True, capture_output=True, check=False,
        )

    def test_historical_loguru_and_imageio_fail_fresh_core_floors(self):
        self.distribution("passage", "1.0.0", ("loguru>=0.8.0", "imageio>=2.38.2"))
        self.distribution("loguru", "0.7.3")
        self.distribution("imageio", "2.38.1")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        for detail in ("loguru>=0.8.0", "0.7.3", "imageio>=2.38.2", "2.38.1"):
            self.assertIn(detail, result.stderr)

    def test_matching_installed_versions_pass(self):
        self.distribution("passage", "1.0.0", ("loguru>=0.8.0", "imageio>=2.38.2"), ">=3.11.5")
        self.distribution("loguru", "0.8.0")
        self.distribution("imageio", "2.38.2")
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 active requirements", result.stdout)

    def test_missing_default_requirement_fails(self):
        self.distribution("passage", "1.0.0", ("passage-guard-missing-test>=1",))
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("passage-guard-missing-test>=1", result.stderr)
        self.assertIn("missing", result.stderr.lower())

    def test_extra_only_missing_requirement_is_ignored(self):
        self.distribution("passage", "1.0.0", ('passage-guard-missing-test>=1; extra == "torch"',))
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_markers_use_current_interpreter(self):
        self.distribution("passage", "1.0.0", (
            'passage-guard-missing-test>=1; python_version < "3.0"',
            'loguru>=0.8.0; python_version >= "3.0" and extra == ""',
        ))
        self.distribution("loguru", "0.7.3")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("loguru>=0.8.0", result.stderr)
        self.assertNotIn("passage-guard-missing-test", result.stderr)

    def test_incompatible_requires_python_fails(self):
        self.distribution("passage", "1.0.0", requires_python="<3")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Requires-Python", result.stderr)
        self.assertIn("<3", result.stderr)

    def test_direct_url_default_requirement_fails_closed(self):
        self.distribution("passage", "1.0.0", (
            "loguru @ https://guard-user:guard-secret@example.invalid/loguru.whl?token=guard-token",
        ))
        self.distribution("loguru", "0.8.0")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("direct URL", result.stderr)
        self.assertIn("loguru", result.stderr)
        for private_detail in ("https://", "guard-user", "guard-secret", "guard-token"):
            self.assertNotIn(private_detail, result.stderr)

    def test_unrelated_nvidia_wheel_tag_is_not_checked(self):
        self.distribution("passage", "1.0.0", ("loguru>=0.8.0",))
        self.distribution("loguru", "0.8.0")
        nvidia = self.distribution("nvidia-cublas", "13.0.0")
        (nvidia / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: historical-nvidia\nRoot-Is-Purelib: false\n"
            "Tag: py3-none-manylinux2014_sbsa\n"
        )
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_malformed_core_requirement_fails_closed(self):
        self.distribution("passage", "1.0.0", ("loguru=>0.8.0",))
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Requires-Dist", result.stderr)

    def test_same_version_stale_installed_metadata_cannot_hide_wheel_floors(self):
        installed = self.distribution("passage", "1.0.0", ("loguru>=0.8.0",))
        (installed / "METADATA").write_text(
            "Metadata-Version: 2.3\nName: passage\nVersion: 1.0.0\nRequires-Dist: loguru>=0.7.3\n\n"
        )
        self.distribution("loguru", "0.7.3")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Requires-Dist", result.stderr)
        self.assertIn("differs", result.stderr)

    def test_installed_core_version_must_match_wheel(self):
        installed = self.distribution("passage", "1.0.0")
        (installed / "METADATA").write_text("Metadata-Version: 2.3\nName: passage\nVersion: 9.0.0\n\n")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Version", result.stderr)
        self.assertIn("differs", result.stderr)

    def test_wheel_with_multiple_core_metadata_files_fails_closed(self):
        self.distribution("passage", "1.0.0")
        with zipfile.ZipFile(self.wheel, "a") as wheel:
            wheel.writestr("another-1.0.0.dist-info/METADATA", "Name: another\nVersion: 1.0.0\n\n")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one", result.stderr)
        self.assertIn("METADATA", result.stderr)

    def test_non_passage_wheel_cannot_validate_installed_core(self):
        self.distribution("passage", "1.0.0")
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            wheel.writestr("another-1.0.0.dist-info/METADATA", "Name: another\nVersion: 1.0.0\n\n")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must identify passage", result.stderr)

    def test_prepared_kit_carries_executable_guard(self):
        self.distribution("passage", "1.0.0", ("loguru>=0.8.0",))
        self.distribution("loguru", "0.7.3")
        runtime = self.root / "runtime"
        (runtime / "pyvenv.cfg").write_text("home = /unavailable-arm64-python\n")
        context = self.root / "context"
        result = subprocess.run(
            [sys.executable, str(PREPARE), "--runtime-env", str(runtime),
             "--core-wheel", str(self.wheel), "--output", str(context)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_guard(context / VERIFIER.name)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("loguru>=0.8.0", result.stderr)
        self.assertIn("0.7.3", result.stderr)


if __name__ == "__main__":
    unittest.main()
