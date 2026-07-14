# Local Data

Place VoxCeleb metadata, HDF5 datasets, and extracted audio under this directory when running the
training or evaluation scripts.

Common paths used by the example configs are:

```text
data/SpeakerDataVox2.hdf5
data/SpeakerDataVox1.hdf5
data/vox2/train_list.txt
data/vox2/test_list.txt
data/vox2/voxceleb2/
data/vox1/voxceleb1/
data/musan_split/
data/RIRS_NOISES/simulated_rirs/
```

The VoxCeleb dataset and generated HDF5 files are intentionally not committed to this repository.

Build an HDF5 file from a list and extracted audio with:

```bash
python Create_h5_dataset/create_new_Dataset.py build-hdf5 \
  --list-path data/vox2/train_list.txt \
  --audio-root data/vox2/voxceleb2 \
  --output-file data/SpeakerDataVox2.hdf5
```
