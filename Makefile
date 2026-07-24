.PHONY: help setup lint format-check typecheck test smoke check precommit

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
		'  precommit     Run pre-commit on all files.'

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
