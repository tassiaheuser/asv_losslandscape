"""Build HDF5 and NumPy datasets from VoxCeleb-style audio lists."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

DEFAULT_CLIP_SAMPLES = 63_488


@dataclass(frozen=True)
class DatasetItem:
    audio_path: Path
    speaker_label: int


class AudioDataset:
    """Dataset backed by `speaker_id relative/path.wav` list entries."""

    def __init__(self, list_path: Path, audio_root: Path):
        self.list_path = list_path
        self.audio_root = audio_root

        if not list_path.is_file():
            raise FileNotFoundError(f"List file not found: {list_path}")
        if not audio_root.is_dir():
            raise FileNotFoundError(f"Audio root directory not found: {audio_root}")

        parsed_lines: list[tuple[str, str]] = []
        for line_number, raw_line in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid entry in {list_path}:{line_number}; expected "
                    "`speaker_id relative/path.wav`."
                )
            parsed_lines.append((fields[0], fields[1]))

        if not parsed_lines:
            raise ValueError(f"No audio entries found in {list_path}")

        speaker_to_index = {
            speaker_id: index
            for index, speaker_id in enumerate(sorted({item[0] for item in parsed_lines}))
        }
        self.items = [
            DatasetItem(audio_root / relative_path, speaker_to_index[speaker_id])
            for speaker_id, relative_path in parsed_lines
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        item = self.items[index]
        audio, _sample_rate = sf.read(item.audio_path, dtype="float32")
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}: {item.audio_path}")
        return audio, item.speaker_label


def create_list(dataset_root: Path, output_file: Path) -> None:
    """Create a list from audio stored as `speaker/video/file.wav`."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    wav_files = sorted(dataset_root.glob("*/*/*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found below {dataset_root}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for wav_file in wav_files:
            speaker_id = wav_file.parts[-3]
            handle.write(f"{speaker_id} {wav_file.relative_to(dataset_root)}\n")


def create_speaker_index(dataset: AudioDataset) -> dict[int, list[int]]:
    speaker_to_indices: defaultdict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(dataset.items):
        speaker_to_indices[item.speaker_label].append(index)
    return dict(speaker_to_indices)


def trim_or_split_audio(
    audio: np.ndarray, clip_samples: int, split_long_audio: bool
) -> list[np.ndarray]:
    if split_long_audio and audio.size >= 2 * clip_samples:
        return [audio[:clip_samples], audio[clip_samples : 2 * clip_samples]]
    return [audio[:clip_samples]]


def fixed_length_audio(audio: np.ndarray, clip_samples: int) -> np.ndarray:
    """Trim or zero-pad audio for a rectangular NumPy array."""
    result = np.zeros(clip_samples, dtype=np.float32)
    length = min(audio.size, clip_samples)
    result[:length] = audio[:length]
    return result


def save_audio_lengths(dataset: AudioDataset, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lengths = [dataset[index][0].size for index in tqdm(range(len(dataset)))]
    np.save(output_file, np.asarray(lengths, dtype=np.int64))


def save_to_hdf5(
    dataset: AudioDataset,
    output_file: Path,
    clip_samples: int = DEFAULT_CLIP_SAMPLES,
    split_long_audio: bool = False,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    speaker_index = create_speaker_index(dataset)

    with h5py.File(output_file, "w") as handle, tqdm(
        total=len(dataset), desc="Writing HDF5"
    ) as progress:
        for speaker_label, item_indices in speaker_index.items():
            group = handle.create_group(f"speaker_{speaker_label}")
            utterance_count = 0
            for item_index in item_indices:
                audio, label = dataset[item_index]
                if label != speaker_label:
                    raise ValueError(f"Speaker mismatch: expected {speaker_label}, got {label}")
                for segment in trim_or_split_audio(audio, clip_samples, split_long_audio):
                    group.create_dataset(f"utterance_{utterance_count}", data=segment)
                    utterance_count += 1
                progress.update(1)
            group.attrs["num_utterances"] = utterance_count


def dual_save(dataset: AudioDataset, output_dir: Path, clip_samples: int) -> None:
    """Save HDF5 plus fixed-width label and audio NumPy arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_index = create_speaker_index(dataset)
    labels: list[int] = []
    audio_segments: list[np.ndarray] = []

    with h5py.File(output_dir / "speaker_data.hdf5", "w") as handle, tqdm(
        total=len(dataset), desc="Writing outputs"
    ) as progress:
        for speaker_label, item_indices in speaker_index.items():
            group = handle.create_group(f"speaker_{speaker_label}")
            for utterance_index, item_index in enumerate(item_indices):
                audio, label = dataset[item_index]
                if label != speaker_label:
                    raise ValueError(f"Speaker mismatch: expected {speaker_label}, got {label}")
                clipped = audio[:clip_samples]
                group.create_dataset(f"utterance_{utterance_index}", data=clipped)
                labels.append(label)
                audio_segments.append(fixed_length_audio(audio, clip_samples))
                progress.update(1)
            group.attrs["num_utterances"] = len(item_indices)

    np.save(output_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_dir / "audio_data.npy", np.stack(audio_segments))


def find_min_max(dataset: AudioDataset) -> tuple[int, int]:
    lengths = [dataset[index][0].size for index in tqdm(range(len(dataset)))]
    minimum, maximum = min(lengths), max(lengths)
    print(f"Minimum length: {minimum}")
    print(f"Maximum length: {maximum}")
    return minimum, maximum


def plot_utterance_histogram(hdf5_path: Path, output_file: Path) -> None:
    import matplotlib.pyplot as plt

    with h5py.File(hdf5_path, "r") as handle:
        counts = [int(group.attrs["num_utterances"]) for group in handle.values()]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.hist(counts, bins=50)
    axis.set(xlabel="Utterances per speaker", ylabel="Speaker count")
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list-path", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("create-list", help="Create a speaker/audio list.")
    list_parser.add_argument("--dataset-root", type=Path, required=True)
    list_parser.add_argument("--output-file", type=Path, required=True)

    hdf5_parser = commands.add_parser("build-hdf5", help="Build an HDF5 dataset.")
    add_dataset_args(hdf5_parser)
    hdf5_parser.add_argument("--output-file", type=Path, required=True)
    hdf5_parser.add_argument("--clip-samples", type=int, default=DEFAULT_CLIP_SAMPLES)
    hdf5_parser.add_argument("--split-long-audio", action="store_true")

    dual_parser = commands.add_parser("dual-save", help="Build HDF5 and NumPy datasets.")
    add_dataset_args(dual_parser)
    dual_parser.add_argument("--output-dir", type=Path, required=True)
    dual_parser.add_argument("--clip-samples", type=int, default=DEFAULT_CLIP_SAMPLES)

    lengths_parser = commands.add_parser("audio-lengths", help="Save audio lengths to NumPy.")
    add_dataset_args(lengths_parser)
    lengths_parser.add_argument("--output-file", type=Path, required=True)

    minmax_parser = commands.add_parser("find-min-max", help="Print minimum/maximum lengths.")
    add_dataset_args(minmax_parser)

    histogram_parser = commands.add_parser("plot-hist", help="Plot an HDF5 speaker histogram.")
    histogram_parser.add_argument("--hdf5-path", type=Path, required=True)
    histogram_parser.add_argument("--output-file", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "create-list":
        create_list(args.dataset_root, args.output_file)
        return
    if args.command == "plot-hist":
        plot_utterance_histogram(args.hdf5_path, args.output_file)
        return

    dataset = AudioDataset(args.list_path, args.audio_root)
    if args.command == "build-hdf5":
        save_to_hdf5(dataset, args.output_file, args.clip_samples, args.split_long_audio)
    elif args.command == "dual-save":
        dual_save(dataset, args.output_dir, args.clip_samples)
    elif args.command == "audio-lengths":
        save_audio_lengths(dataset, args.output_file)
    elif args.command == "find-min-max":
        find_min_max(dataset)


if __name__ == "__main__":
    main()
