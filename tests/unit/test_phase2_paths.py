"""Unit tests for Phase 2 safe-path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from protoem_ct.data import (
    DatasetPathEscapeError,
    DuplicateResolvedPathError,
    InvalidDatasetRootError,
    InvalidRelativeDatasetPathError,
    MissingDatasetPathError,
    NonRegularDatasetFileError,
    normalize_safe_relative_posix_path,
    require_unique_resolved_paths,
    resolve_existing_path_beneath_root,
    resolve_regular_file_beneath_root,
    validate_explicit_dataset_root,
    validate_explicit_external_output_root,
)


def test_valid_absolute_dataset_root_is_accepted(tmp_path: Path) -> None:
    assert validate_explicit_dataset_root(tmp_path) == tmp_path.resolve()


def test_relative_dataset_root_rejected_even_when_it_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_dataset_root(Path("dataset"))


def test_missing_and_regular_file_dataset_roots_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_dataset_root(tmp_path / "missing")

    file_root = tmp_path / "root-file.txt"
    file_root.write_text("placeholder\n", encoding="utf-8")
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_dataset_root(file_root)


def test_dataset_root_result_is_independent_of_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    before = validate_explicit_dataset_root(root)

    monkeypatch.chdir(other)

    assert validate_explicit_dataset_root(root) == before


@pytest.mark.parametrize(
    "value",
    [
        "images/case-001.txt",
        "Images/Subdir/Case_001-file.txt",
    ],
)
def test_valid_nested_posix_relative_paths_are_accepted(value: str) -> None:
    assert normalize_safe_relative_posix_path(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "/absolute/file.txt",
        "../file.txt",
        "images/../file.txt",
        "images/./file.txt",
        "./images/file.txt",
        "images//file.txt",
        "images\\file.txt",
        "file://images/file.txt",
        "s3://bucket/file.txt",
        "C:/images/file.txt",
        " images/file.txt",
        "images/file.txt ",
        "images/\x00file.txt",
        "images/\nfile.txt",
    ],
)
def test_unsafe_relative_paths_are_rejected(value: str) -> None:
    with pytest.raises(InvalidRelativeDatasetPathError):
        normalize_safe_relative_posix_path(value)


def test_safe_resolution_accepts_regular_file_and_existing_directory(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "case-001.txt"
    image.write_text("image placeholder\n", encoding="utf-8")

    assert resolve_existing_path_beneath_root(root, "images") == image_dir.resolve()
    assert resolve_regular_file_beneath_root(root, "images/case-001.txt") == image.resolve()


def test_safe_resolution_rejects_missing_and_directory_for_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)

    with pytest.raises(MissingDatasetPathError):
        resolve_existing_path_beneath_root(root, "images/missing.txt")

    with pytest.raises(NonRegularDatasetFileError):
        resolve_regular_file_beneath_root(root, "images")


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unsupported")
def test_symlink_escape_rejected_and_internal_symlink_accepted(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    internal_target = root / "inside.txt"
    internal_target.write_text("inside\n", encoding="utf-8")

    escape_link = root / "escape.txt"
    internal_link = root / "internal-link.txt"
    try:
        escape_link.symlink_to(outside)
        internal_link.symlink_to(internal_target)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    with pytest.raises(DatasetPathEscapeError):
        resolve_regular_file_beneath_root(root, "escape.txt")

    assert resolve_regular_file_beneath_root(root, "internal-link.txt") == internal_target.resolve()


def test_resolution_is_independent_of_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    file_path = root / "case.txt"
    file_path.write_text("placeholder\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()

    monkeypatch.chdir(other)

    assert resolve_regular_file_beneath_root(root, "case.txt") == file_path.resolve()


def test_output_root_validation_accepts_nonexistent_existing_and_sibling_roots(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    existing_output = tmp_path / "output"
    existing_output.mkdir()
    missing_output = tmp_path / "missing-output"
    sibling_output = tmp_path / "sibling-output"

    assert validate_explicit_external_output_root(missing_output) == missing_output
    assert validate_explicit_external_output_root(existing_output) == existing_output.resolve()
    assert (
        validate_explicit_external_output_root(
            sibling_output,
            forbidden_roots=(dataset_root,),
        )
        == sibling_output
    )
    assert not missing_output.exists()
    assert not sibling_output.exists()


def test_output_root_rejects_file_relative_and_dataset_containment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_file = tmp_path / "output.txt"
    output_file.write_text("placeholder\n", encoding="utf-8")
    inside_dataset = dataset_root / "generated"
    parent_output = tmp_path

    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_external_output_root(output_file)
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_external_output_root(Path("relative-output"))
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_external_output_root(dataset_root, forbidden_roots=(dataset_root,))
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_external_output_root(inside_dataset, forbidden_roots=(dataset_root,))
    with pytest.raises(InvalidDatasetRootError):
        validate_explicit_external_output_root(parent_output, forbidden_roots=(dataset_root,))


def test_require_unique_resolved_paths_rejects_duplicates_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    shared = root / "shared.txt"
    shared.write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(DuplicateResolvedPathError) as first_error:
        require_unique_resolved_paths((("z-record", shared), ("a-record", shared)))

    with pytest.raises(DuplicateResolvedPathError) as second_error:
        require_unique_resolved_paths((("a-record", shared), ("z-record", shared)))

    assert str(first_error.value) == str(second_error.value)
    assert "a-record + z-record" in str(first_error.value)
