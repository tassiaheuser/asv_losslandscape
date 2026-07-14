# Local Checkpoints

Put pretrained VoxCeleb checkpoint files in this directory when running the example configs.

Checkpoint files are intentionally not committed to this repository because they can be large and
may have separate redistribution restrictions. The example configs use paths such as:

```text
voxceleb/trained_net/weights_om/Res2NeXt6s8g/Vox1-O/model000000130.model
```

Create the matching local directory structure or edit `model_file` in the YAML config you run.
