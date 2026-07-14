from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import unittest

import yaml

from voxceleb.download_data.dataprep import HTTPSOnlyRedirectHandler, download_file, safe_target


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositoryQualityTest(unittest.TestCase):
    def test_downloader_rejects_insecure_urls_and_archive_traversal(self) -> None:
        with self.assertRaises(ValueError):
            download_file("http://example.invalid/archive.zip", Path("unused.zip"))
        with self.assertRaises(ValueError):
            HTTPSOnlyRedirectHandler().redirect_request(
                None, None, 302, "Found", {}, "http://example.invalid/archive.zip"
            )
        with self.assertRaises(ValueError):
            safe_target(REPOSITORY_ROOT / "data", "../outside")

    def test_download_tools_contain_no_insecure_transport(self) -> None:
        paths = [REPOSITORY_ROOT / "voxceleb" / "download_data"]
        for root in paths:
            for path in root.rglob("*"):
                if path.suffix in {".py", ".sh", ".txt"}:
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("http://", text, path)
                    self.assertNotIn("--no-check-certificate", text, path)

    def test_voxceleb_config_keys_are_known_cli_options(self) -> None:
        source = (REPOSITORY_ROOT / "voxceleb" / "trainSpeakerNet.py").read_text()
        known_options = set(re.findall(r"add_argument\(\s*['\"]--([\w-]+)", source))
        for path in (REPOSITORY_ROOT / "voxceleb" / "configs").rglob("*"):
            if path.suffix not in {".yaml", ".yml"}:
                continue
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self.assertEqual(set(config) - known_options, set(), path)

    def test_model_snapshots_keep_the_public_factory_interface(self) -> None:
        model_names = {
            "RawNet3.py",
            "Res2NeXt6s8g.py",
            "Res2Net8s.py",
            "ResNeXt6g.py",
            "ResNetSE34L.py",
            "ResNetSE34L_Bott.py",
            "ResNetSE34V2.py",
        }
        roots = [
            REPOSITORY_ROOT / "voxceleb" / "models",
            REPOSITORY_ROOT / "loss-landscape" / "voxceleb" / "model",
        ]
        for root in roots:
            for name in model_names:
                path = root / name
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                self.assertIn("MainModel", functions, path)

    def test_notebooks_have_no_saved_execution_state(self) -> None:
        for path in REPOSITORY_ROOT.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook.get("cells", []):
                self.assertFalse(cell.get("outputs", []), path)
                self.assertIsNone(cell.get("execution_count"), path)


if __name__ == "__main__":
    unittest.main()
