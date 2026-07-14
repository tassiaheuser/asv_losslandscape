# Contributing

Small corrections, portability improvements, and reproducibility fixes are welcome. Please avoid
committing VoxCeleb data, credentials, checkpoints, generated surfaces, or experiment logs.

Before opening a pull request:

```bash
python -m pip install h5py numpy PyYAML soundfile tqdm
python -m compileall -q loss-landscape voxceleb examples tests
python -m unittest discover -s tests -v
git diff --check
```

Changes to model definitions should identify the affected checkpoint format and whether the
training and loss-landscape snapshots must both change. New experiments should include a config,
random seed, environment information, and artifact checksums.
