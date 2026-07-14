#!/usr/bin/env python3
"""Download, extract, and convert VoxCeleb data and augmentation files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler
from zipfile import ZipFile

from tqdm import tqdm

VOXCELEB_ARCHIVES = (
    "vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b",
    "vox2_dev_aac_parta* vox2_dev_aac.zip bbc063c46078a602ca71605645c2a402",
)
PUBLIC_AUGMENTATION_ARCHIVES = (
    "https://www.openslr.org/resources/28/rirs_noises.zip "
    "e6f48e257286e05de56413b4779d8ffb",
    "https://www.openslr.org/resources/17/musan.tar.gz "
    "0c472d4fc0c5141eca47ad1ffeb2a7df",
)


class HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    """Prevent an HTTPS download from being redirected to plaintext HTTP."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise ValueError(f"Refusing insecure redirect URL: {new_url}")
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_file: Path) -> None:
    if urlsplit(url).scheme != "https":
        raise ValueError(f"Refusing insecure download URL: {url}")
    opener = build_opener(HTTPSOnlyRedirectHandler())
    with opener.open(url) as response, output_file.open("wb") as destination:
        shutil.copyfileobj(response, destination, length=1024 * 1024)


def download(save_path: Path, lines: list[str]) -> None:
    for line in lines:
        url, expected_md5 = line.split()
        output_file = save_path / Path(urlsplit(url).path).name
        print(f"Downloading {output_file.name}")
        download_file(url, output_file)
        actual_md5 = md5(output_file)
        if actual_md5 != expected_md5:
            output_file.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum failed for {output_file.name}: expected {expected_md5}, "
                f"got {actual_md5}."
            )


def concatenate(save_path: Path, lines: list[str]) -> None:
    for line in lines:
        input_pattern, output_name, expected_md5 = line.split()
        input_files = sorted(save_path.glob(input_pattern))
        if not input_files:
            raise FileNotFoundError(f"No files match {save_path / input_pattern}")
        output_file = save_path / output_name
        with output_file.open("wb") as destination:
            for input_file in input_files:
                with input_file.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        if md5(output_file) != expected_md5:
            output_file.unlink(missing_ok=True)
            raise ValueError(f"Checksum failed for concatenated file {output_file}")
        for input_file in input_files:
            input_file.unlink()


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes extraction directory: {member_name}")
    return target


def full_extract(save_path: Path, archive: Path) -> None:
    print(f"Extracting {archive}")
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                safe_target(save_path, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported archive member: {member.name}")
            tar.extractall(save_path)
    elif archive.suffix == ".zip":
        with ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                safe_target(save_path, member.filename)
            zip_file.extractall(save_path)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def partial_extract(save_path: Path, archive: Path, prefixes: tuple[str, ...]) -> None:
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            if member.filename.startswith(prefixes):
                safe_target(save_path, member.filename)
                zip_file.extract(member, save_path)


def convert(save_path: Path) -> None:
    files = sorted(save_path.glob("*/*/*.m4a"))
    for input_file in tqdm(files, desc="Converting AAC to WAV"):
        output_file = input_file.with_suffix(".wav")
        if output_file.exists():
            continue
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_file), "-ac", "1", "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", str(output_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def split_musan(save_path: Path) -> None:
    from scipy.io import wavfile

    files = sorted(save_path.glob("musan/*/*/*.wav"))
    audio_length, audio_stride = 16000 * 5, 16000 * 3
    for input_file in tqdm(files, desc="Splitting MUSAN"):
        sample_rate, audio = wavfile.read(input_file)
        output_dir = Path(str(input_file.with_suffix("")).replace("/musan/", "/musan_split/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(audio) - audio_length, audio_stride):
            wavfile.write(output_dir / f"{start / sample_rate:05.0f}.wav", sample_rate, audio[start : start + audio_length])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-path", "--save_path", type=Path, default=Path("data"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Download and prepare public MUSAN/RIRS augmentation data over HTTPS.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)
    if args.augment:
        download(args.save_path, list(PUBLIC_AUGMENTATION_ARCHIVES))
        partial_extract(
            args.save_path,
            args.save_path / "rirs_noises.zip",
            ("RIRS_NOISES/simulated_rirs/mediumroom", "RIRS_NOISES/simulated_rirs/smallroom"),
        )
        full_extract(args.save_path, args.save_path / "musan.tar.gz")
        split_musan(args.save_path)

    if args.extract:
        concatenate(args.save_path, list(VOXCELEB_ARCHIVES))
        for line in VOXCELEB_ARCHIVES:
            full_extract(args.save_path, args.save_path / line.split()[1])
        shutil.move(str(args.save_path / "dev" / "aac"), args.save_path / "aac")
        shutil.rmtree(args.save_path / "dev")
        shutil.move(str(args.save_path / "wav"), args.save_path / "voxceleb1")
        shutil.move(str(args.save_path / "aac"), args.save_path / "voxceleb2")

    if args.convert:
        convert(args.save_path)


if __name__ == "__main__":
    main()
