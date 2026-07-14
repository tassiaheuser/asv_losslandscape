from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import soundfile as sf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "voxceleb" / "Create_h5_dataset" / "create_new_Dataset.py"
SPEC = importlib.util.spec_from_file_location("dataset_builder", MODULE_PATH)
assert SPEC and SPEC.loader
dataset_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dataset_builder
SPEC.loader.exec_module(dataset_builder)


class DatasetBuilderTest(unittest.TestCase):
    def test_build_hdf5_and_fixed_width_numpy_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_root = root / "audio"
            paths = [audio_root / "id1" / "video1" / "a.wav", audio_root / "id2" / "video1" / "b.wav"]
            for path, length in zip(paths, (12, 25)):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.linspace(-0.5, 0.5, length, dtype=np.float32), 16_000)

            list_path = root / "train_list.txt"
            dataset_builder.create_list(audio_root, list_path)
            dataset = dataset_builder.AudioDataset(list_path, audio_root)
            self.assertEqual(len(dataset), 2)

            hdf5_path = root / "dataset.hdf5"
            dataset_builder.save_to_hdf5(dataset, hdf5_path, clip_samples=16)
            with h5py.File(hdf5_path) as handle:
                self.assertEqual(handle["speaker_0"].attrs["num_utterances"], 1)
                self.assertEqual(handle["speaker_1"].attrs["num_utterances"], 1)

            output_dir = root / "dual"
            dataset_builder.dual_save(dataset, output_dir, clip_samples=16)
            audio = np.load(output_dir / "audio_data.npy")
            self.assertEqual(audio.shape, (2, 16))
            self.assertTrue(np.all(audio[0, 12:] == 0))


if __name__ == "__main__":
    unittest.main()
