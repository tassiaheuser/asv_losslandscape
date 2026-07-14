# Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization

This repository contains PyTorch code used for the master thesis
"Evaluating ResNet-based Speaker Verification Systems through Loss Landscape Visualization".

> Tassia Heuser

The code is based on [tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape),
which implements:

> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
> [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913). NeurIPS, 2018.

The original loss-landscape workflow is kept, with extensions for VoxCeleb speaker-verification
models, HDF5-based data loading, surface extension/refinement, and thesis-specific plotting.

## Repository Structure

```text
configs/              Example YAML configs and SLURM submit scripts.
notebooks/            Exploratory plotting notebooks.
voxceleb/loss/        Loss functions used by VoxCeleb models.
voxceleb/model/       Model definitions.
voxceleb/trained_net/ Expected location for local checkpoint files; ignored by git.
dataloader_toplevel.py
model_loader_toplevel.py
plot_surface.py       Main script for generating loss landscapes.
plot_trajectory.py    Projects training trajectories onto landscape directions.
plot_1D.py, plot_2D.py, plotting.py
utils.py              YAML config helpers and small utilities.
```

## Setup

Python 3.10 or newer is recommended. Create an environment and install the direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches your CUDA setup if the default package
resolution does not match your system.

## Data and Checkpoints

Large datasets, generated surfaces, and model checkpoints are not included in this repository.
Place local files under the paths used by the example configs, or edit the config values:

```text
data/SpeakerDataVox2.hdf5
data/vox2/train_list.txt
data/vox2/test_list.txt
data/vox2/voxceleb2/
data/vox1/voxceleb_test/
voxceleb/trained_net/
Output/
```

`voxceleb/trained_net/`, `data/`, and `Output/` are intentionally treated as local artifact
locations. Do not commit pretrained checkpoints or VoxCeleb data unless you have the right to
redistribute them.

The example configs record the checkpoint layout used for the thesis. Checkpoint download
instructions will be added separately; until then, update `model_file` to a compatible local model.

## Usage

Run a loss-landscape job from a YAML config:

```bash
python plot_surface.py --config configs/run/resnext6s8g/resnext6s8g_train.yaml
```

Project an optimization trajectory. Pass `--dir_file` to reuse existing directions; omit it to
create PCA directions from the checkpoint sequence:

```bash
python plot_trajectory.py \
  --model_folder voxceleb/trained_net/example_run \
  --model res2next6s8g \
  --max_epoch 300 \
  --save_epoch 1
```

Convert a generated HDF5 surface to VTP for ParaView:

```bash
python h52vtp.py --surf_file Output/example/surface.h5
```

## Configs and SLURM

Files in `configs/run/` are examples. They now use repository-relative paths, so clone-specific
paths should be set by editing the YAML or passing command-line arguments.

The SLURM scripts run from `${SLURM_SUBMIT_DIR}`. If you want them to activate a conda
environment, submit from a shell where conda is already initialized, or set:

```bash
export LOSS_LANDSCAPE_CONDA_ENV=lola2
```

## Main Additions

- VoxCeleb model-loading and loss-function support.
- Faster HDF5-based VoxCeleb data loading.
- Surface resolution increase and range extension for existing `.h5` landscape files.
- Plotting utilities for 1D, 2D, 3D, and trajectory visualizations.

## License

This repository keeps the MIT license from the original loss-landscape project. See
[`LICENSE`](LICENSE) for details.
