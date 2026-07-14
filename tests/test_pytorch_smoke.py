from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional test dependency")
class PyTorchModelSmokeTest(unittest.TestCase):
    def test_resnet_model_constructs_and_runs_on_cpu(self) -> None:
        script = textwrap.dedent(
            """
            import torch
            from models.ResNetSE34V2 import MainModel

            model = MainModel(nOut=16, n_mels=40, encoder_type="SAP", log_input=True).eval()
            with torch.no_grad():
                output = model(torch.randn(1, 16_000))
            if output.shape != (1, 16):
                raise AssertionError(f"Unexpected output shape: {output.shape}")
            """
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT / "voxceleb",
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
