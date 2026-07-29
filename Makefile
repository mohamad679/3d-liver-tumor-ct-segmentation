.PHONY: help setup lint format-check typecheck test smoke check precommit phase2-synthetic baseline baseline-test

help:
	@printf '%s\n' \
		'Available Phase 0 commands:' \
		'  help          Print the available Phase 0 commands.' \
		'  setup         Run uv sync for Python 3.11 and all dependency groups.' \
		'  lint          Run Ruff checks, format check, and mypy on src.' \
		'  format-check  Run Ruff format --check .' \
		'  typecheck     Run mypy src.' \
		'  test          Run pytest -q.' \
		'  smoke         Run smoke tests from tests/smoke.' \
		'  check         Run lint and test.' \
		'  precommit     Run pre-commit on all files.' \
		'  phase2-synthetic Run the synthetic-only Phase 2 DAG.' \
		'  baseline      Run the synthetic Phase 3 Gate 3 orchestration in the isolated baseline environment.' \
		'  baseline-test Run the selected Phase 3 tests in the isolated baseline environment.'

setup:
	uv sync --python 3.11 --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

smoke:
	uv run pytest -q tests/smoke

check: lint test

precommit:
	uv run pre-commit run --all-files

phase2-synthetic:
	uv run snakemake --cores 1 phase2_synthetic_all --configfile configs/phase2_synthetic.yaml --config phase2_synthetic_git_commit="$$(git rev-parse HEAD)"

baseline:
	@test -n "$(PHASE3_BASELINE_OUTPUT_ROOT)" || (echo "PHASE3_BASELINE_OUTPUT_ROOT is required" && exit 1)
	@case "$(PHASE3_BASELINE_OUTPUT_ROOT)" in /*) ;; *) echo "PHASE3_BASELINE_OUTPUT_ROOT must be an absolute path"; exit 1 ;; esac
	uv run --project environments/phase3-baselines/intel-macos-cpu --locked --no-sync protoem-ct run-phase3-gate3-synthetic --output-root "$(PHASE3_BASELINE_OUTPUT_ROOT)" --git-commit "$$(git rev-parse HEAD)" --start-timestamp "2026-07-29T00:00:00Z" --end-timestamp "2026-07-29T01:00:00Z"

baseline-test:
	env -u VIRTUAL_ENV uv run --project environments/phase3-baselines/intel-macos-cpu --locked --no-sync python -m pytest -q -x \
		tests/unit/test_torch_runtime.py \
		tests/unit/test_phase3_mlflow_local.py \
		tests/unit/test_monai_segresnet.py \
		tests/unit/test_nnunet_tiny.py \
		tests/unit/test_gate3_report.py \
		tests/unit/test_nnunet_wrapper.py \
		tests/integration/test_phase3_gate3_cli.py
