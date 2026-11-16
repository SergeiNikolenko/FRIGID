"""Real tar round-trip coverage for portable historical evidence."""
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "packager", Path(__file__).resolve().parents[1] / "scripts/package_benchmark_artifacts.py"
)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class ArchiveTest(unittest.TestCase):
    def test_preserves_bytes_and_pax_links_without_external_placeholders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "original.tar.gz", root / "portable.tar.gz"
            file_path = "docs/FRIGID_project/result.txt"
            with tarfile.open(source, "w:gz", format=tarfile.PAX_FORMAT) as archive:
                entry = tarfile.TarInfo(file_path)
                entry.size = 7
                archive.addfile(entry, io.BytesIO(b"result\n"))
                for name, target in [
                    ("internal", "/old/worktree/" + file_path),
                    ("external", "/old/server/missing_dataset"),
                ]:
                    entry = tarfile.TarInfo("docs/FRIGID_project/" + name)
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = target
                    entry.pax_headers["linkpath"] = target
                    archive.addfile(entry)
            original_hash = packager.sha256(source)
            result = packager.package(source, output, "historical-commit")
            self.assertEqual(packager.sha256(source), original_hash)
            self.assertEqual(result["regular_files"], 1)
            self.assertEqual(result["relative_internal_links"], 1)
            self.assertEqual(result["external_reference_count"], 1)
            with tarfile.open(output) as archive:
                self.assertEqual(archive.extractfile(file_path).read(), b"result\n")
                self.assertEqual(archive.getmember("docs/FRIGID_project/internal").linkname, "result.txt")
                self.assertNotIn("docs/FRIGID_project/external", archive.getnames())
                manifest = json.load(archive.extractfile("docs/FRIGID_project/archive_manifest.json"))
                self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(b"result\n").hexdigest())
                self.assertFalse(manifest["dataset_complete"])
            with self.assertRaises(FileExistsError):
                packager.package(source, output, "historical-commit")

    def test_rejects_traversing_archive_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bad.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                archive.addfile(tarfile.TarInfo("docs/FRIGID_project/../../outside"))
            with self.assertRaisesRegex(ValueError, "Unexpected archive path"):
                packager.package(source, Path(temp) / "out.tar.gz", "historical-commit")


if __name__ == "__main__":
    unittest.main()
