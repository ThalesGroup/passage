"""Checks that constructing a private kit cannot edit its source environment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name("prepare_ml_context.py")


class SnapshotTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("prepare_ml_context", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_snapshot_preserves_source_and_removes_only_editable_passage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            for name in ("module.py", "__editable__.passage-1.0.0.pth", "__editable___passage_1_0_0_finder.py", "_virtualenv.pth"):
                (source / name).write_text(name)
            manifest, removed = self.module().snapshot_site_packages(source, target)
            self.assertTrue((source / "__editable__.passage-1.0.0.pth").is_file())
            self.assertFalse((target / "__editable__.passage-1.0.0.pth").exists())
            self.assertEqual(set(manifest), {"module.py", "_virtualenv.pth"})
            self.assertEqual(len(removed), 2)
            self.assertEqual((source / "module.py").stat().st_ino, (target / "module.py").stat().st_ino)
            self.module().verify_files(target, manifest)
            (target / "new.py").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "file set"):
                self.module().verify_files(target, manifest)

    def test_snapshot_rejects_symlink_before_materializing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "escape").symlink_to(root / "outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.module().snapshot_site_packages(source, root / "target")


if __name__ == "__main__":
    unittest.main()
