"""Generate deterministic synthetic LiTS-style Phase 2 fixture files.

The generated HMAC key is fixed, synthetic-only test material. It is deliberately unsuitable for
real data, is written outside the synthetic dataset root, and must never be tracked or printed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from snakemake.exceptions import WorkflowError

SYNTHETIC_ID_KEY = b"protoem-ct-phase2-synthetic-hmac-key-v1-0000"
SHAPE = (6, 7, 5)
SPACING = (1.25, 1.5, 2.5)
AFFINE = np.diag([*SPACING, 1.0]).astype(np.float64)

Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]


def required_config_value(config: dict[str, Any], name: str) -> str:
    """Return a nonempty config value, using a safe placeholder for lint/Phase 2 parsing."""
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        if "--lint" in sys.argv or _is_phase2_synthetic_target():
            if name == "generated_root":
                return str(Path.cwd() / ".snakemake-lint-placeholder")
            return f"lint-placeholder-{name}"
        raise WorkflowError(f"Missing required Snakemake --config value: {name}")
    return value


def _is_phase2_synthetic_target() -> bool:
    """Return whether this invocation requested the Phase 2 synthetic target family."""
    return any(arg == "phase2_synthetic_all" or arg.startswith("phase2_") for arg in sys.argv)


def validate_child_path(root: Path, path: Path) -> Path:
    """Require path to remain beneath root after lexical path construction."""
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkflowError(f"Workflow output escapes generated_root: {path}") from exc
    return path


def phase2_config(config: dict[str, Any], name: str) -> Any:
    """Return a required Phase 2 synthetic config value."""
    value = config.get(name)
    if value is None:
        raise WorkflowError(f"Missing Phase 2 synthetic config value: {name}")
    return value


def phase2_nonempty_config(config: dict[str, Any], name: str) -> str:
    """Return a required nonempty Phase 2 synthetic string config value."""
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        if "--lint" in sys.argv:
            return f"lint-placeholder-{name}"
        raise WorkflowError(f"Missing required Snakemake --config value: {name}")
    return value


def phase2_bool_text(value: object) -> str:
    """Return the Typer boolean flag for a Phase 2 recursive-layout setting."""
    return "--recursive" if value else "--no-recursive"


def phase2_child_path(root: Path, *parts: str) -> Path:
    """Build a generated-root-contained Phase 2 output path."""
    path = root.joinpath(*parts)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkflowError(f"Phase 2 synthetic output escapes generated root: {path}") from exc
    return path


def phase2_run_cli(argv: list[object], log_path: object) -> None:
    """Run one public CLI command with argv-list execution and sanitized stdout logging."""
    result = subprocess.run(
        [str(item) for item in argv],
        capture_output=True,
        text=True,
        check=False,
    )
    log_file = Path(str(log_path))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(result.stdout, encoding="utf-8", newline="\n")
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"command failed: {argv[0]}"
        raise WorkflowError(message)


def phase2_generate_fixture(output: Any, log: Any, params: Any) -> None:
    """Run the synthetic fixture generator."""
    result = subprocess.run(
        [
            sys.executable,
            str(params.script),
            "--dataset-root",
            str(params.dataset_root),
            "--control-root",
            str(Path(str(output.marker)).parent),
            "--marker",
            str(output.marker),
            "--case-count",
            str(params.case_count),
            "--git-commit",
            str(params.git_commit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = Path(str(log))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8", newline="\n")
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "fixture generation failed"
        raise WorkflowError(message)


def phase2_inventory_lits(output: Any, params: Any) -> None:
    """Run sanitized LiTS-style inventory dry-run."""
    phase2_run_cli(
        [
            "protoem-ct",
            "inventory-lits",
            "--dataset-root",
            params.dataset_root,
            "--image-pattern",
            params.image_pattern,
            "--label-pattern",
            params.label_pattern,
            params.recursive,
            "--image-prefix",
            params.image_prefix,
            "--label-prefix",
            params.label_prefix,
        ],
        output.log,
    )


def phase2_build_lits_manifest(output: Any, params: Any) -> None:
    """Run anonymous LiTS-style manifest publication."""
    phase2_run_cli(
        [
            "protoem-ct",
            "build-lits-manifest",
            "--dataset-root",
            params.dataset_root,
            "--image-pattern",
            params.image_pattern,
            "--label-pattern",
            params.label_pattern,
            params.recursive,
            "--image-prefix",
            params.image_prefix,
            "--label-prefix",
            params.label_prefix,
            "--dataset-id",
            params.dataset_id,
            "--project-namespace",
            params.project_namespace,
            "--patient-id-prefix",
            params.patient_id_prefix,
            "--case-id-prefix",
            params.case_id_prefix,
            "--digest-length",
            str(params.digest_length),
            "--id-key-file",
            params.key_file,
            "--output",
            output.artifact,
            "--git-commit",
            params.git_commit,
            "--generated-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def phase2_build_development_split(input: Any, output: Any, params: Any) -> None:
    """Run deterministic patient-level split generation."""
    phase2_run_cli(
        [
            "protoem-ct",
            "build-development-split",
            "--manifest",
            input.manifest,
            "--output",
            output.artifact,
            "--policy-version",
            params.policy_version,
            "--split-seed",
            str(params.split_seed),
            "--train-patient-count",
            str(params.train_count),
            "--validation-patient-count",
            str(params.validation_count),
            "--internal-test-patient-count",
            str(params.internal_test_count),
            "--git-commit",
            params.git_commit,
            "--generated-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def phase2_geometry_label_qa(input: Any, output: Any, params: Any) -> None:
    """Run geometry and label QA."""
    argv: list[object] = [
        "protoem-ct",
        "qa-geometry-labels",
        "--manifest",
        input.manifest,
        "--split",
        input.split,
        "--dataset-root",
        params.dataset_root,
        "--output",
        output.artifact,
    ]
    for label in params.allowed_labels:
        argv.extend(["--allowed-label-value", str(label)])
    argv.extend(
        [
            "--tumor-label-value",
            str(params.tumor_label),
            "--affine-tolerance",
            str(params.affine_tolerance),
            "--git-commit",
            params.git_commit,
            "--created-at-utc",
            params.created_at_utc,
        ]
    )
    phase2_run_cli(argv, output.log)


def phase2_summarize_lesions(input: Any, output: Any, params: Any) -> None:
    """Run deterministic lesion connected-component summaries."""
    phase2_run_cli(
        [
            "protoem-ct",
            "summarize-lesions",
            "--manifest",
            input.manifest,
            "--geometry-qa-artifact",
            input.geometry,
            "--dataset-root",
            params.dataset_root,
            "--output",
            output.artifact,
            "--connectivity",
            str(params.connectivity),
            "--tumor-label-value",
            str(params.tumor_label),
            "--git-commit",
            params.git_commit,
            "--created-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def phase2_summarize_development_data(input: Any, output: Any, params: Any) -> None:
    """Run deterministic development-data summaries."""
    phase2_run_cli(
        [
            "protoem-ct",
            "summarize-development-data",
            "--manifest",
            input.manifest,
            "--geometry-qa-artifact",
            input.geometry,
            "--lesion-artifact",
            input.lesion,
            "--dataset-root",
            params.dataset_root,
            "--output",
            output.artifact,
            "--histogram-min",
            str(params.histogram_min),
            "--histogram-max",
            str(params.histogram_max),
            "--histogram-bin-count",
            str(params.histogram_bin_count),
            "--git-commit",
            params.git_commit,
            "--created-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def phase2_build_development_qa_report(input: Any, output: Any, params: Any) -> None:
    """Run final DevelopmentQaArtifact JSON report assembly."""
    phase2_run_cli(
        [
            "protoem-ct",
            "build-development-qa-report",
            "--manifest",
            input.manifest,
            "--split",
            input.split,
            "--geometry-qa-artifact",
            input.geometry,
            "--lesion-artifact",
            input.lesion,
            "--development-summary-artifact",
            input.summary,
            "--output",
            output.artifact,
            "--report-contract-version",
            params.contract_version,
            "--git-commit",
            params.git_commit,
            "--created-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def phase2_audit_development_leakage(input: Any, output: Any, params: Any) -> None:
    """Run deterministic development leakage audit."""
    phase2_run_cli(
        [
            "protoem-ct",
            "audit-development-leakage",
            "--manifest",
            input.manifest,
            "--split",
            input.split,
            "--output",
            output.artifact,
            "--audit-contract-version",
            params.contract_version,
            "--git-commit",
            params.git_commit,
            "--created-at-utc",
            params.created_at_utc,
        ],
        output.log,
    )


def main() -> None:
    """Generate a deterministic four-case synthetic Phase 2 fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--case-count", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    if args.case_count != 4:
        raise SystemExit("Phase 2 synthetic fixture requires exactly four cases")
    if not args.git_commit.strip():
        raise SystemExit("phase2_synthetic_git_commit must be supplied before fixture generation")

    dataset_root = args.dataset_root.resolve(strict=False)
    control_root = args.control_root.resolve(strict=False)
    if dataset_root == control_root or _is_relative_to(control_root, dataset_root):
        raise SystemExit("synthetic HMAC key must be outside the synthetic dataset root")

    volume_root = dataset_root / "volumes"
    label_root = dataset_root / "segmentations"
    key_path = control_root / "phase2_synthetic_hmac.key"
    volume_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)

    for index in range(4):
        image, label = _case_arrays(index)
        _write_nifti_if_absent_or_equal(volume_root / f"volume-{index}.nii", image)
        _write_nifti_if_absent_or_equal(label_root / f"segmentation-{index}.nii", label)
    _write_bytes_if_absent_or_equal(key_path, SYNTHETIC_ID_KEY)
    _write_text_if_absent_or_equal(args.marker, "phase2 synthetic fixture generated\n")


def _case_arrays(index: int) -> tuple[Float32Array, UInt8Array]:
    grid = np.indices(SHAPE, dtype=np.float32)
    image = ((index + 1) * 10.0 + grid[0] * 1.25 + grid[1] * 0.5 - grid[2] * 0.75).astype(
        np.float32
    )
    label = np.zeros(SHAPE, dtype=np.uint8)
    if index == 0:
        label[1:3, 1:3, 1:3] = 1
    elif index == 1:
        label[1:3, 1:3, 1:2] = 1
        label[4:6, 4:6, 3:5] = 1
    elif index == 2:
        label[2:5, 2:5, 1:4] = 1
    elif index == 3:
        label[0, 0, 0] = 0
    else:  # pragma: no cover - guarded by caller
        raise ValueError(index)
    return image, label


def _write_nifti_if_absent_or_equal(path: Path, array: Float32Array | UInt8Array) -> None:
    image = nib.Nifti1Image(array, AFFINE)
    image.header.set_data_dtype(array.dtype)
    image.header.set_xyzt_units("mm")
    image.set_qform(AFFINE, code=1)
    image.set_sform(AFFINE, code=1)
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    nib.save(image, str(temp_path))
    try:
        _publish_temp_if_absent_or_equal(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _write_bytes_if_absent_or_equal(path: Path, content: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(content)
    _publish_temp_if_absent_or_equal(temp_path, path)


def _write_text_if_absent_or_equal(path: Path, content: str) -> None:
    _write_bytes_if_absent_or_equal(path, content.encode("utf-8"))


def _publish_temp_if_absent_or_equal(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        if final_path.read_bytes() != temp_path.read_bytes():
            raise SystemExit(f"existing synthetic fixture file differs: {final_path.name}")
        temp_path.unlink()
        return
    os.replace(temp_path, final_path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
