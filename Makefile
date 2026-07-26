.PHONY: help setup lint format-check typecheck test smoke check precommit phase2-synthetic

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
		'  phase2-synthetic Run the synthetic-only Phase 2 DAG.'

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
