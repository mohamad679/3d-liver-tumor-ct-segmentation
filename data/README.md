# Data Handling

Real medical datasets must never be committed to this repository.

LiTS/MSD-style development data and 3D-IRCADb external data must remain outside Git.

Manifests must contain anonymous identifiers and only relative paths or environment-resolved paths.

Model weights, predictions, DICOM files, NIfTI medical volumes, credentials, and secrets are prohibited from version control.

Only tiny generated synthetic smoke-test data may be used inside this repository, subject to later Phase 0 implementation.

