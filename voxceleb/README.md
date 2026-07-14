# VoxCeleb Speaker Verification Trainer

This repository contains PyTorch code used to train and evaluate automatic speaker verification
models for the master thesis "Evaluating ResNet-based Speaker Verification Systems through Loss
Landscape Visualization".

> Tassia Heuser

The code is based on [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer),
which implements models and recipes from the VoxCeleb speaker-recognition literature.

## Repository Structure

```text
configs/              Example YAML configs for training and evaluation.
Create_h5_dataset/    Utilities for building HDF5 datasets from VoxCeleb audio.
download_data/        Dataset download/extraction helper scripts.
loss/                 Loss functions.
models/               Model definitions.
models/weights/       Local checkpoint location; ignored except for README.md.
notebooks/            Exploratory analysis notebooks.
optimizer/            Optimizer wrappers.
scheduler/            Learning-rate schedulers.
data/                 Local data location; ignored except for README.md.
SpeakerNet.py         Training and evaluation loops.
trainSpeakerNet.py    Main CLI for training and evaluation.
tuneThreshold.py      EER and MinDCF utilities.
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch and torchaudio builds that match your CUDA version if the
default packages are not appropriate for your machine.

## Data and Checkpoints

VoxCeleb audio, generated HDF5 datasets, experiment outputs, and pretrained checkpoints are not
included. Place them under local paths such as:

```text
data/SpeakerDataVox2.hdf5
data/SpeakerDataVox1.hdf5
data/vox2/train_list.txt
data/vox2/test_list.txt
data/vox2/voxceleb2/
data/vox1/voxceleb1/
models/weights/ResNetSE34V2/baseline_v2_smproto.model
exps/
```

See `data/README.md` and `models/weights/README.md` for the expected local layout.

## Usage

Train from a config:

```bash
python trainSpeakerNet.py --config configs/ResNetSE34v2_AAMsoftmax.yml
```

Evaluate from a config:

```bash
python trainSpeakerNet.py --config configs/eval/ResNetSE34v2_AAMSoftmax.yml --eval
```

Create an HDF5 dataset from local VoxCeleb lists/audio:

```bash
python Create_h5_dataset/create_new_Dataset.py build-hdf5 \
  --list-path data/vox2/train_list.txt \
  --audio-root data/vox2/voxceleb2 \
  --output-file data/SpeakerDataVox2.hdf5
```

Run `python Create_h5_dataset/create_new_Dataset.py --help` to see commands for generating list
files, exporting NumPy arrays, inspecting audio lengths, and plotting HDF5 statistics.

Request VoxCeleb access through the
[official dataset page](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), then place the authorized
archives or archive parts under `data/`. Authenticated downloads are intentionally not automated.
To concatenate and safely extract manually obtained archives, run:

```bash
python download_data/dataprep.py --save-path data --extract
```

Public MUSAN and RIRS augmentation data can be downloaded over verified HTTPS with:

```bash
python download_data/dataprep.py --save-path data --augment
```

## Reproducing experiments

The example configs record the data and checkpoint layout used for the thesis. Checkpoints are not
currently distributed with this repository. Until checkpoint download instructions are added, use
your own compatible checkpoint or remove `initial_model` when starting a suitable configuration
from scratch. The large default batch sizes may also need adjustment for your hardware.

## SLURM

`sbatch.sh` and the scripts in `download_data/` are examples. They use repository-relative paths
and write logs to `slurm_logs/`.

If your cluster uses conda, submit from a shell where conda is initialized and optionally set:

```bash
export VOXCELEB_CONDA_ENV=lola2
```

## Main Additions

- Weights & Biases logging support.
- HDF5-based data loading for larger VoxCeleb runs.
- Train-set evaluation utilities.
- Additional cosine and constant learning-rate schedulers.
- Resume support through stored optimizer and scheduler state.

## Citation

Please cite the upstream VoxCeleb trainer work if you use this code:

```bibtex
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

```bibtex
@inproceedings{kwon2021ins,
  title={The ins and outs of speaker recognition: lessons from {VoxSRC} 2020},
  author={Kwon, Yoohwan and Heo, Hee Soo and Lee, Bong-Jin and Chung, Joon Son},
  booktitle={Proc. ICASSP},
  year={2021}
}
```

```bibtex
@inproceedings{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2022}
}
```

## License

This repository keeps the upstream MIT license and notices. See `LICENSE.md` and `NOTICE.md`.
