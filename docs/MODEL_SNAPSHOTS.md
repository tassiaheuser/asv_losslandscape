# Model implementation snapshots

`voxceleb/models/` is the authoritative training implementation. The similarly named files under
`loss-landscape/voxceleb/model/` are analysis snapshots adapted to the loss-landscape loader and
historical checkpoints. They are intentionally not imported across component boundaries because
the two tools use different module layouts and were developed from different upstream projects.

Several snapshots differ in defaults or architecture details. Do not replace one copy with another
without verifying checkpoint state-dict compatibility. CI checks that every shared model family
continues to expose the common `MainModel` factory expected by both loaders.

When modifying a shared model:

1. Decide whether the change applies to training, landscape analysis, or both.
2. Run the CPU model smoke test.
3. Test loading a representative checkpoint for every affected snapshot.
4. Record intentional divergence here and in the experiment metadata.
