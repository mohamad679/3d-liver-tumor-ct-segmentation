"""Command-line interface for ProtoEM-CT."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from protoem_ct.data.synthetic import (
    SyntheticDataError,
    create_synthetic_dataset,
    load_synthetic_config,
)
from protoem_ct.data.validation import NiftiValidationError, validate_nifti_pair

app = typer.Typer(help="ProtoEM-CT command-line tools.")


@app.callback()
def main() -> None:
    """Run ProtoEM-CT command-line tools."""


@app.command("create-data")
def create_data(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the synthetic data OmegaConf YAML file.",
        ),
    ] = Path("configs/data/synthetic.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Empty or nonexistent root under which generated data will be written.",
        ),
    ] = ...,  # type: ignore[assignment]
    git_commit: Annotated[
        str,
        typer.Option(
            "--git-commit",
            help="Explicit Git commit recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
    created_at_utc: Annotated[
        str,
        typer.Option(
            "--created-at-utc",
            help="Explicit UTC creation timestamp recorded in the manifest.",
        ),
    ] = ...,  # type: ignore[assignment]
) -> None:
    """Create deterministic Phase 1 synthetic NIfTI inputs and manifest."""
    try:
        config = load_synthetic_config(config_path)
        result = create_synthetic_dataset(
            config,
            output_root=output_root,
            git_commit=git_commit,
            created_at_utc=created_at_utc,
        )
    except SyntheticDataError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("create-data success")
    typer.echo(f"case count: {len(result.manifest.case_ids)}")
    typer.echo(f"manifest path: {result.manifest_relative_path.as_posix()}")
    typer.echo(f"config hash: {result.config_hash}")
    typer.echo(f"manifest hash: {result.manifest_hash}")


@app.command("validate-pair")
def validate_pair(
    image_path: Annotated[Path, typer.Argument(help="Path to the NIfTI image file.")],
    label_path: Annotated[Path, typer.Argument(help="Path to the NIfTI label file.")],
    affine_tolerance: Annotated[
        float,
        typer.Option(
            "--affine-tolerance",
            help="Absolute tolerance for affine matrix comparison.",
        ),
    ] = 1e-5,
) -> None:
    """Validate a NIfTI image and binary label pair."""
    try:
        result = validate_nifti_pair(
            image_path,
            label_path,
            affine_tolerance=affine_tolerance,
        )
    except NiftiValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo("validation success")
    typer.echo(f"shape: {result.shape}")
    typer.echo(f"label values: {list(result.label_values)}")
