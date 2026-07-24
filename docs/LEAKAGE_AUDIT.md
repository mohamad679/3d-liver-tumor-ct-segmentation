# Leakage Audit

## Phase 0 Placeholder

This document is a Phase 0 placeholder. No leakage audit has been executed, and no audit outcome should be inferred from its presence.

Planned checks:

- Patient overlap across train and validation sets
- Patient overlap across train and test sets
- Patient overlap across validation and test sets
- Patient overlap between support and query sets
- Patient overlap between support sets and any train, validation, or test sets
- Patient overlap between query sets and any train, validation, or test sets

External labels cannot be used for preprocessing fitting, tuning, threshold selection, checkpoint selection, or support construction.

LiTS and MSD Task03 Liver are not independent cohorts.

No audit has passed at this stage.
