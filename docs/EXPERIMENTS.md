# Experiment map

This table maps the main experiment families to their versioned configuration. Checkpoints and
large result files are external artifacts and must be recorded with `checkpoints.example.yml`.

| Experiment | Model | Configuration | Required external artifact | Output family |
| --- | --- | --- | --- | --- |
| VoxCeleb2 AAM-softmax training | ResNetSE34V2 | `voxceleb/configs/ResNetSE34v2_AAMsoftmax.yml` | Initial ResNetSE34V2 checkpoint | `voxceleb/exps/ResNetSE34Lv2_AAMSoftmax/` |
| VoxCeleb2 softmax training | ResNetSE34V2 | `voxceleb/configs/ResNetSE34v2_softmax.yml` | Initial ResNetSE34V2 checkpoint | Configured `save_path` |
| Res2NeXt6s8g landscape | Res2NeXt6s8g | `loss-landscape/configs/run/resnext6s8g/resnext6s8g_train.yaml` | Trained model checkpoint and VoxCeleb2 HDF5 | `loss-landscape/Output/resnext6s8g_train/` |
| Balanced Res2NeXt6s8g landscape | Res2NeXt6s8g | `loss-landscape/configs/run/resnext6s8g_balancedTraining/` | Balanced-training checkpoint and VoxCeleb2 HDF5 | `loss-landscape/Output/resnext6s8g_balancedTraining/` |
| Res2Net8s landscape refinement | Res2Net8s | `loss-landscape/configs/run/res2net8s_train/` | Trained model checkpoint, base surface, and VoxCeleb2 HDF5 | `loss-landscape/Output/res2net8s_train/` |
| Softmax surface extension | ResNetSE34V2 | `loss-landscape/configs/run/softmaxExt/vox2_v2_Softmax_Ext.yaml` | Softmax checkpoint and base surface | Configured `out_dir` |

Reported metrics and final figure identifiers should be added alongside checkpoint metadata when
the external artifacts are recovered. This avoids publishing unverifiable numbers.
