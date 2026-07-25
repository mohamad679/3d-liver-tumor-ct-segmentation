# Acceptance Checklist

## Gate 0

- [x] Standard repository structure
- [x] Python 3.11 configuration
- [x] uv and `pyproject.toml`
- [x] ruff
- [x] mypy
- [x] pytest
- [x] hypothesis
- [x] pre-commit
- [x] Makefile
- [x] CPU-only GitHub Actions
- [x] Tiny synthetic NIfTI CT and tumor-mask fixtures
- [x] `protoem-ct validate-pair` CLI
- [x] Tests for shape mismatch
- [x] Tests for affine mismatch
- [x] Tests for invalid labels
- [x] Tests for NaNs
- [x] Lint passes
- [x] Typing passes
- [x] Tests pass
- [x] Smoke test passes

### Gate 0 Evidence

Gate 0 status: PASSED locally

- Python: 3.11.9
- `uv sync --frozen --python 3.11 --all-groups`: PASS
- `uv lock --check`: PASS
- `make lint`: PASS
- `make test`: PASS, `24 passed`
- `make smoke`: PASS, `1 passed`
- `uv run mypy src tests`: PASS
- `uv run pre-commit validate-config`: PASS
- `uv run pre-commit run --all-files`: PASS
- `uv run protoem-ct --help`: PASS
- `uv run protoem-ct validate-pair --help`: PASS
- No tracked NIfTI, DICOM, model weights, checkpoints, predictions, credentials, or generated binary artifacts.
- Synthetic NIfTI files are created dynamically under pytest temporary directories only.
- GitHub-hosted CI is not claimed as passed until it runs on GitHub.

## Gate 1

- [ ] Complete synthetic DAG runs from a clean generated-artifact state
- [ ] Every stage has explicit inputs and outputs
- [ ] Artifacts use documented schemas
- [ ] Report is generated only from persisted JSON
- [ ] Config, manifest, and Git metadata are recorded
- [ ] Repeated runs are deterministic where expected
- [ ] Integration test passes
- [ ] Lint passes
- [ ] Typing passes
- [ ] Full tests pass
- [ ] Smoke tests pass

## Gate 2

- [ ] Pending definition.

## Gate 3

- [ ] Pending definition.

## Gate 4

- [ ] Pending definition.

## Gate 5

- [ ] Pending definition.

## Gate 6

- [ ] Pending definition.

## Gate 7

- [ ] Pending definition.

## Gate 8

- [ ] Pending definition.
