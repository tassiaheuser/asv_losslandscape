# Reproducing the workflow

The full experiments require authorized VoxCeleb data and compatible checkpoints. The commands
below separate steps that anyone can run from steps that require those external artifacts.

## 1. Validate the checkout

From the repository root:

```bash
python -m pip install h5py numpy PyYAML soundfile tqdm
python -m unittest discover -s tests -v
```

## 2. Run the unrestricted toy example

```bash
python -m pip install numpy matplotlib
python examples/toy_loss_landscape.py
```

This writes `docs/images/toy_loss_landscape.png` and verifies the basic two-direction landscape
idea without using research data.

## 3. Prepare VoxCeleb data

1. Request access through the [official VoxCeleb page](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
2. Place manually obtained archives under `voxceleb/data/` and follow the official terms.
3. Extract archives and convert audio as needed:

   ```bash
   cd voxceleb
   python download_data/dataprep.py --save-path data --extract --convert
   ```

4. Create the HDF5 training data:

   ```bash
   python Create_h5_dataset/create_new_Dataset.py build-hdf5 \
     --list-path data/vox2/train_list.txt \
     --audio-root data/vox2/voxceleb2 \
     --output-file data/SpeakerDataVox2.hdf5
   ```

## 4. Train or evaluate a model

Install the pinned reference environment and select a config:

```bash
cd voxceleb
python -m pip install -r requirements.txt -c ../constraints/research-environment.txt
python trainSpeakerNet.py --config configs/ResNetSE34v2_AAMsoftmax.yml
```

Configs that specify `initial_model` require the corresponding local checkpoint. Record its source
and SHA-256 digest using `docs/checkpoints.example.yml`.

## 5. Generate a loss landscape

```bash
cd loss-landscape
python -m pip install -r requirements.txt -c ../constraints/research-environment.txt
python plot_surface.py --config configs/run/resnext6s8g/resnext6s8g_train.yaml
```

Update `model_file` and data paths in the selected config. Output HDF5 surfaces and plots are
written below `Output/`, which is intentionally ignored by Git.

## 6. Record provenance

For every reported run, retain the Git commit, config, checkpoint SHA-256 digest, dataset version,
Python version, dependency constraints, CUDA/PyTorch versions, random seed, and hardware.
