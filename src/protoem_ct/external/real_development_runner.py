"""Phase 8 real-development run planning and orchestration scaffold (Substage 2).

This module is a metadata-only, manifest/split-driven planning and orchestration
scaffold for a *future* real MONAI SegResNet development run. It performs no
filesystem access to image, label, prediction, or checkpoint bytes; it never
constructs a model, loads training data, runs inference, computes a real
metric, or trains anything. Every plan produced here is permanently locked to
``pixel_access_state == "not_started"`` and
``execution_release_state == "scaffold_only"``; there is no public API in this
module capable of producing or accepting any other value. Attempting to invoke
the (unimplemented) Substage 3 executor through
:func:`execute_phase8_real_development_run` always raises
:class:`Phase8RealDevelopmentExecutionNotReleasedError`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

from protoem_ct.artifacts import (
    DatasetManifest,
    DevelopmentSplitManifest,
    hash_dataset_manifest,
    hash_development_split,
    phase2_artifact_from_json,
)
from protoem_ct.artifacts.hashing import JsonValue, canonical_json_bytes, sha256_file, sha256_json
from protoem_ct.data._phase2_publication import (
    Phase2PublicationExistingOutputError,
    Phase2PublicationIOError,
    publish_text_no_overwrite,
)
from protoem_ct.data.phase2_paths import (
    InvalidDatasetRootError,
    normalize_safe_relative_posix_path,
    validate_explicit_external_output_root,
)
from protoem_ct.external.internal_evidence import (
    ArtifactReference,
    Phase8FixedCandidateInventory,
    artifact_reference_from_mapping,
    artifact_reference_to_dict,
)

MappingLike: TypeAlias = Mapping[str, object]

PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION: Final[str] = "v1"
PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_SCHEMA_NAME: Final[str] = (
    "phase8_real_development_input_binding"
)
PHASE8_REAL_DEVELOPMENT_RUN_PLAN_SCHEMA_NAME: Final[str] = "phase8_real_development_run_plan"
PHASE8_REAL_DEVELOPMENT_CASE_BINDING_SCHEMA_NAME: Final[str] = (
    "phase8_real_development_case_binding"
)
PHASE8_REAL_DEVELOPMENT_CASE_BINDING_COLLECTION_SCHEMA_NAME: Final[str] = (
    "phase8_real_development_case_binding_collection"
)
PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_SCHEMA_NAME: Final[str] = (
    "phase8_real_development_plan_summary"
)

PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_FILENAME: Final[str] = (
    "phase8_real_development_input_binding.json"
)
PHASE8_REAL_DEVELOPMENT_RUN_PLAN_FILENAME: Final[str] = "phase8_real_development_run_plan.json"
PHASE8_REAL_DEVELOPMENT_CASE_BINDINGS_FILENAME: Final[str] = (
    "phase8_real_development_case_bindings.json"
)
PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME: Final[str] = (
    "phase8_real_development_plan_summary.json"
)

PIXEL_ACCESS_NOT_STARTED: Final[str] = "not_started"
_ALLOWED_PIXEL_ACCESS_STATES: Final[frozenset[str]] = frozenset(
    {"not_started", "tiny_dry_run_pending", "released"}
)
EXECUTION_RELEASE_SCAFFOLD_ONLY: Final[str] = "scaffold_only"
_ALLOWED_EXECUTION_RELEASE_STATES: Final[frozenset[str]] = frozenset(
    {"scaffold_only", "released_for_tiny_dry_run", "released"}
)
_ALLOWED_CASE_BINDING_PARTITIONS: Final[frozenset[str]] = frozenset({"train", "validation"})

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,126}[a-z0-9])?$")
_ZERO_SHA256: Final[str] = "0" * 64
_UNRESOLVED_TOKENS: Final[frozenset[str]] = frozenset(
    {"pending", "placeholder", "tbd", "todo", "unknown", "unresolved"}
)


class Phase8RealDevelopmentRunnerError(ValueError):
    """Base error for the Phase 8 real-development planning scaffold."""


class Phase8RealDevelopmentValidationError(Phase8RealDevelopmentRunnerError):
    """Raised when a planning contract violates its invariants."""


class Phase8RealDevelopmentSerializationError(Phase8RealDevelopmentRunnerError):
    """Raised when a planning mapping cannot be reconstructed strictly."""


class Phase8RealDevelopmentHashError(Phase8RealDevelopmentRunnerError):
    """Raised when a planning self-hash or input file hash does not match."""


class Phase8RealDevelopmentPublicationError(Phase8RealDevelopmentRunnerError):
    """Raised when guarded scaffold-plan publication fails safely."""


class Phase8RealDevelopmentExecutionNotReleasedError(Phase8RealDevelopmentRunnerError):
    """Raised unconditionally: Substage 3 real-development execution is not released."""


# ---------------------------------------------------------------------------
# A. Metadata-only development input binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8RealDevelopmentInputBinding:
    """Metadata-only binding to a verified Phase 2 manifest and development split."""

    schema_name: str
    schema_version: str
    manifest_reference: ArtifactReference
    split_reference: ArtifactReference
    manifest_file_sha256: str
    split_file_sha256: str
    expected_manifest_schema_name: str
    expected_manifest_schema_version: str
    expected_split_schema_name: str
    expected_split_schema_version: str
    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    internal_test_patient_ids: tuple[str, ...]
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    internal_test_case_ids: tuple[str, ...]
    approved_development_artifact_set_identity: str
    no_external_data: bool
    input_binding_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_sha256(self.manifest_file_sha256, field_name="manifest_file_sha256")
        _require_sha256(self.split_file_sha256, field_name="split_file_sha256")
        _require_nonempty_token(
            self.expected_manifest_schema_name,
            field_name="expected_manifest_schema_name",
        )
        _require_nonempty_token(
            self.expected_manifest_schema_version,
            field_name="expected_manifest_schema_version",
        )
        _require_nonempty_token(
            self.expected_split_schema_name,
            field_name="expected_split_schema_name",
        )
        _require_nonempty_token(
            self.expected_split_schema_version,
            field_name="expected_split_schema_version",
        )
        object.__setattr__(self, "train_patient_ids", tuple(self.train_patient_ids))
        object.__setattr__(self, "validation_patient_ids", tuple(self.validation_patient_ids))
        object.__setattr__(self, "internal_test_patient_ids", tuple(self.internal_test_patient_ids))
        object.__setattr__(self, "train_case_ids", tuple(self.train_case_ids))
        object.__setattr__(self, "validation_case_ids", tuple(self.validation_case_ids))
        object.__setattr__(self, "internal_test_case_ids", tuple(self.internal_test_case_ids))
        _require_sorted_unique_ids(self.train_patient_ids, field_name="train_patient_ids")
        _require_sorted_unique_ids(self.validation_patient_ids, field_name="validation_patient_ids")
        _require_sorted_unique_ids(
            self.internal_test_patient_ids, field_name="internal_test_patient_ids"
        )
        _require_sorted_unique_ids(self.train_case_ids, field_name="train_case_ids")
        _require_sorted_unique_ids(self.validation_case_ids, field_name="validation_case_ids")
        _require_sorted_unique_ids(self.internal_test_case_ids, field_name="internal_test_case_ids")
        if not self.train_patient_ids or not self.validation_patient_ids:
            raise Phase8RealDevelopmentValidationError(
                "input binding requires nonempty train and validation partitions."
            )
        _require_disjoint(
            self.train_patient_ids,
            self.validation_patient_ids,
            self.internal_test_patient_ids,
            field_name="patient partitions",
        )
        _require_disjoint(
            self.train_case_ids,
            self.validation_case_ids,
            self.internal_test_case_ids,
            field_name="case partitions",
        )
        _require_identifier(
            self.approved_development_artifact_set_identity,
            field_name="approved_development_artifact_set_identity",
        )
        _require_true(self.no_external_data, field_name="no_external_data")
        _require_self_hash(
            self.input_binding_hash,
            _input_binding_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                manifest_reference=self.manifest_reference,
                split_reference=self.split_reference,
                manifest_file_sha256=self.manifest_file_sha256,
                split_file_sha256=self.split_file_sha256,
                expected_manifest_schema_name=self.expected_manifest_schema_name,
                expected_manifest_schema_version=self.expected_manifest_schema_version,
                expected_split_schema_name=self.expected_split_schema_name,
                expected_split_schema_version=self.expected_split_schema_version,
                train_patient_ids=self.train_patient_ids,
                validation_patient_ids=self.validation_patient_ids,
                internal_test_patient_ids=self.internal_test_patient_ids,
                train_case_ids=self.train_case_ids,
                validation_case_ids=self.validation_case_ids,
                internal_test_case_ids=self.internal_test_case_ids,
                approved_development_artifact_set_identity=(
                    self.approved_development_artifact_set_identity
                ),
                no_external_data=self.no_external_data,
            ),
            field_name="input_binding_hash",
        )


def _input_binding_payload(
    *,
    schema_name: str,
    schema_version: str,
    manifest_reference: ArtifactReference,
    split_reference: ArtifactReference,
    manifest_file_sha256: str,
    split_file_sha256: str,
    expected_manifest_schema_name: str,
    expected_manifest_schema_version: str,
    expected_split_schema_name: str,
    expected_split_schema_version: str,
    train_patient_ids: Sequence[str],
    validation_patient_ids: Sequence[str],
    internal_test_patient_ids: Sequence[str],
    train_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    internal_test_case_ids: Sequence[str],
    approved_development_artifact_set_identity: str,
    no_external_data: bool,
) -> dict[str, JsonValue]:
    return {
        "approved_development_artifact_set_identity": (approved_development_artifact_set_identity),
        "expected_manifest_schema_name": expected_manifest_schema_name,
        "expected_manifest_schema_version": expected_manifest_schema_version,
        "expected_split_schema_name": expected_split_schema_name,
        "expected_split_schema_version": expected_split_schema_version,
        "internal_test_case_ids": list(internal_test_case_ids),
        "internal_test_patient_ids": list(internal_test_patient_ids),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_reference": artifact_reference_to_dict(manifest_reference),
        "no_external_data": no_external_data,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "split_file_sha256": split_file_sha256,
        "split_reference": artifact_reference_to_dict(split_reference),
        "train_case_ids": list(train_case_ids),
        "train_patient_ids": list(train_patient_ids),
        "validation_case_ids": list(validation_case_ids),
        "validation_patient_ids": list(validation_patient_ids),
    }


def phase8_real_development_input_binding_identity_payload(
    binding: Phase8RealDevelopmentInputBinding,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for an input binding."""

    return _input_binding_payload(
        schema_name=binding.schema_name,
        schema_version=binding.schema_version,
        manifest_reference=binding.manifest_reference,
        split_reference=binding.split_reference,
        manifest_file_sha256=binding.manifest_file_sha256,
        split_file_sha256=binding.split_file_sha256,
        expected_manifest_schema_name=binding.expected_manifest_schema_name,
        expected_manifest_schema_version=binding.expected_manifest_schema_version,
        expected_split_schema_name=binding.expected_split_schema_name,
        expected_split_schema_version=binding.expected_split_schema_version,
        train_patient_ids=binding.train_patient_ids,
        validation_patient_ids=binding.validation_patient_ids,
        internal_test_patient_ids=binding.internal_test_patient_ids,
        train_case_ids=binding.train_case_ids,
        validation_case_ids=binding.validation_case_ids,
        internal_test_case_ids=binding.internal_test_case_ids,
        approved_development_artifact_set_identity=(
            binding.approved_development_artifact_set_identity
        ),
        no_external_data=binding.no_external_data,
    )


def hash_phase8_real_development_input_binding(
    binding: Phase8RealDevelopmentInputBinding,
) -> str:
    """Return the canonical self-hash for an input binding."""

    return sha256_json(phase8_real_development_input_binding_identity_payload(binding))


def phase8_real_development_input_binding_to_dict(
    binding: Phase8RealDevelopmentInputBinding,
) -> dict[str, JsonValue]:
    """Convert an input binding to a canonical mapping."""

    payload = phase8_real_development_input_binding_identity_payload(binding)
    payload["input_binding_hash"] = binding.input_binding_hash
    return payload


def phase8_real_development_input_binding_to_json(
    binding: Phase8RealDevelopmentInputBinding,
) -> bytes:
    """Serialize an input binding to canonical JSON bytes."""

    return canonical_json_bytes(phase8_real_development_input_binding_to_dict(binding)) + b"\n"


_INPUT_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "manifest_reference",
        "split_reference",
        "manifest_file_sha256",
        "split_file_sha256",
        "expected_manifest_schema_name",
        "expected_manifest_schema_version",
        "expected_split_schema_name",
        "expected_split_schema_version",
        "train_patient_ids",
        "validation_patient_ids",
        "internal_test_patient_ids",
        "train_case_ids",
        "validation_case_ids",
        "internal_test_case_ids",
        "approved_development_artifact_set_identity",
        "no_external_data",
        "input_binding_hash",
    }
)


def phase8_real_development_input_binding_from_mapping(
    mapping: MappingLike,
) -> Phase8RealDevelopmentInputBinding:
    """Reconstruct an input binding from a strict mapping."""

    _require_exact_fields(
        mapping, _INPUT_BINDING_FIELDS, object_name="Phase8RealDevelopmentInputBinding"
    )
    return Phase8RealDevelopmentInputBinding(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        manifest_reference=artifact_reference_from_mapping(
            _expect_mapping(mapping["manifest_reference"], field_name="manifest_reference")
        ),
        split_reference=artifact_reference_from_mapping(
            _expect_mapping(mapping["split_reference"], field_name="split_reference")
        ),
        manifest_file_sha256=_expect_string(
            mapping["manifest_file_sha256"], field_name="manifest_file_sha256"
        ),
        split_file_sha256=_expect_string(
            mapping["split_file_sha256"], field_name="split_file_sha256"
        ),
        expected_manifest_schema_name=_expect_string(
            mapping["expected_manifest_schema_name"],
            field_name="expected_manifest_schema_name",
        ),
        expected_manifest_schema_version=_expect_string(
            mapping["expected_manifest_schema_version"],
            field_name="expected_manifest_schema_version",
        ),
        expected_split_schema_name=_expect_string(
            mapping["expected_split_schema_name"], field_name="expected_split_schema_name"
        ),
        expected_split_schema_version=_expect_string(
            mapping["expected_split_schema_version"],
            field_name="expected_split_schema_version",
        ),
        train_patient_ids=_expect_string_tuple(
            mapping["train_patient_ids"], field_name="train_patient_ids"
        ),
        validation_patient_ids=_expect_string_tuple(
            mapping["validation_patient_ids"], field_name="validation_patient_ids"
        ),
        internal_test_patient_ids=_expect_string_tuple(
            mapping["internal_test_patient_ids"], field_name="internal_test_patient_ids"
        ),
        train_case_ids=_expect_string_tuple(mapping["train_case_ids"], field_name="train_case_ids"),
        validation_case_ids=_expect_string_tuple(
            mapping["validation_case_ids"], field_name="validation_case_ids"
        ),
        internal_test_case_ids=_expect_string_tuple(
            mapping["internal_test_case_ids"], field_name="internal_test_case_ids"
        ),
        approved_development_artifact_set_identity=_expect_string(
            mapping["approved_development_artifact_set_identity"],
            field_name="approved_development_artifact_set_identity",
        ),
        no_external_data=_expect_bool(mapping["no_external_data"], field_name="no_external_data"),
        input_binding_hash=_expect_string(
            mapping["input_binding_hash"], field_name="input_binding_hash"
        ),
    )


def _validate_single_file_path(path: Path, *, field_name: str) -> Path:
    if str(path) == "":
        raise Phase8RealDevelopmentValidationError(f"{field_name} must not be empty.")
    if not path.is_absolute():
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be an absolute path.")
    if path.is_symlink():
        raise Phase8RealDevelopmentValidationError(f"{field_name} must not be a symlink.")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise Phase8RealDevelopmentValidationError(f"{field_name} does not exist.") from exc
    except OSError as exc:
        raise Phase8RealDevelopmentValidationError(f"{field_name} cannot be resolved.") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be a regular file.")
    return resolved


def build_phase8_real_development_input_binding(
    *,
    manifest_path: Path,
    split_path: Path,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    approved_development_artifact_set_identity: str,
) -> Phase8RealDevelopmentInputBinding:
    """Bind a Phase 2 dataset manifest and development split by verified hash only.

    Reads exactly two explicit JSON files (never image or label bytes), verifies
    each against a caller-declared SHA-256 expectation, parses them with the
    strict Phase 2 reconstructors, re-verifies their embedded content hashes,
    and cross-checks partition membership and disjointness.
    """

    resolved_manifest_path = _validate_single_file_path(manifest_path, field_name="manifest_path")
    resolved_split_path = _validate_single_file_path(split_path, field_name="split_path")

    _require_sha256(expected_manifest_sha256, field_name="expected_manifest_sha256")
    _require_sha256(expected_split_sha256, field_name="expected_split_sha256")

    observed_manifest_sha256 = sha256_file(resolved_manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise Phase8RealDevelopmentHashError("manifest file SHA-256 does not match expectation.")
    observed_split_sha256 = sha256_file(resolved_split_path)
    if observed_split_sha256 != expected_split_sha256:
        raise Phase8RealDevelopmentHashError("split file SHA-256 does not match expectation.")

    manifest_text = resolved_manifest_path.read_text(encoding="utf-8")
    split_text = resolved_split_path.read_text(encoding="utf-8")
    try:
        manifest = phase2_artifact_from_json(manifest_text, DatasetManifest)
    except Exception as exc:
        raise Phase8RealDevelopmentSerializationError(
            "failed to parse dataset manifest JSON."
        ) from exc
    try:
        split = phase2_artifact_from_json(split_text, DevelopmentSplitManifest)
    except Exception as exc:
        raise Phase8RealDevelopmentSerializationError(
            "failed to parse development split JSON."
        ) from exc

    if hash_dataset_manifest(manifest) != manifest.manifest_hash:
        raise Phase8RealDevelopmentHashError("manifest embedded hash does not match content.")
    if hash_development_split(split) != split.split_hash:
        raise Phase8RealDevelopmentHashError("split embedded hash does not match content.")
    if split.source_manifest_hash != manifest.manifest_hash:
        raise Phase8RealDevelopmentValidationError(
            "split source_manifest_hash does not match the bound manifest."
        )

    patient_ids_by_partition: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "internal_test": set(),
    }
    case_ids_by_partition: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "internal_test": set(),
    }
    for assignment in split.assignments:
        patient_ids_by_partition[assignment.partition].add(assignment.anonymous_patient_id)
        case_ids_by_partition[assignment.partition].add(assignment.anonymous_case_id)

    train_patient_ids = tuple(sorted(patient_ids_by_partition["train"]))
    validation_patient_ids = tuple(sorted(patient_ids_by_partition["validation"]))
    internal_test_patient_ids = tuple(sorted(patient_ids_by_partition["internal_test"]))
    train_case_ids = tuple(sorted(case_ids_by_partition["train"]))
    validation_case_ids = tuple(sorted(case_ids_by_partition["validation"]))
    internal_test_case_ids = tuple(sorted(case_ids_by_partition["internal_test"]))

    manifest_reference = ArtifactReference(
        schema_name=manifest.manifest_type,
        schema_version=f"v{manifest.schema_version}",
        artifact_hash=manifest.manifest_hash,
        artifact_role="development_manifest",
    )
    split_reference = ArtifactReference(
        schema_name=split.manifest_type,
        schema_version=f"v{split.schema_version}",
        artifact_hash=split.split_hash,
        artifact_role="development_split",
    )

    payload = _input_binding_payload(
        schema_name=PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        manifest_reference=manifest_reference,
        split_reference=split_reference,
        manifest_file_sha256=observed_manifest_sha256,
        split_file_sha256=observed_split_sha256,
        expected_manifest_schema_name=manifest.manifest_type,
        expected_manifest_schema_version=manifest.schema_version,
        expected_split_schema_name=split.manifest_type,
        expected_split_schema_version=split.schema_version,
        train_patient_ids=train_patient_ids,
        validation_patient_ids=validation_patient_ids,
        internal_test_patient_ids=internal_test_patient_ids,
        train_case_ids=train_case_ids,
        validation_case_ids=validation_case_ids,
        internal_test_case_ids=internal_test_case_ids,
        approved_development_artifact_set_identity=approved_development_artifact_set_identity,
        no_external_data=True,
    )
    return Phase8RealDevelopmentInputBinding(
        schema_name=PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        manifest_reference=manifest_reference,
        split_reference=split_reference,
        manifest_file_sha256=observed_manifest_sha256,
        split_file_sha256=observed_split_sha256,
        expected_manifest_schema_name=manifest.manifest_type,
        expected_manifest_schema_version=manifest.schema_version,
        expected_split_schema_name=split.manifest_type,
        expected_split_schema_version=split.schema_version,
        train_patient_ids=train_patient_ids,
        validation_patient_ids=validation_patient_ids,
        internal_test_patient_ids=internal_test_patient_ids,
        train_case_ids=train_case_ids,
        validation_case_ids=validation_case_ids,
        internal_test_case_ids=internal_test_case_ids,
        approved_development_artifact_set_identity=approved_development_artifact_set_identity,
        no_external_data=True,
        input_binding_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# B. Real-development run plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8RealDevelopmentRunPlan:
    """Immutable, scaffold-only plan for a future real-development training run."""

    schema_name: str
    schema_version: str
    input_binding_hash: str
    fixed_candidate_inventory_hash: str
    candidate_id: str
    model_family: str
    training_adaptation_mode: str
    training_config_reference: ArtifactReference
    preprocessing_decision_reference: ArtifactReference
    fixed_seeds: tuple[int, ...]
    intended_train_patient_ids: tuple[str, ...]
    intended_validation_patient_ids: tuple[str, ...]
    internal_test_excluded: bool
    expected_checkpoint_metadata_schema_name: str
    expected_checkpoint_metadata_schema_version: str
    expected_validation_evidence_schema_name: str
    expected_validation_evidence_schema_version: str
    expected_output_artifact_names: tuple[str, ...]
    no_external_data: bool
    pixel_access_state: str
    execution_release_state: str
    run_plan_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_REAL_DEVELOPMENT_RUN_PLAN_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_sha256(self.input_binding_hash, field_name="input_binding_hash")
        _require_sha256(
            self.fixed_candidate_inventory_hash,
            field_name="fixed_candidate_inventory_hash",
        )
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_identifier(self.model_family, field_name="model_family")
        _require_identifier(self.training_adaptation_mode, field_name="training_adaptation_mode")
        object.__setattr__(self, "fixed_seeds", tuple(self.fixed_seeds))
        if not self.fixed_seeds:
            raise Phase8RealDevelopmentValidationError("fixed_seeds must not be empty.")
        for index, seed in enumerate(self.fixed_seeds):
            _require_nonnegative_int(seed, field_name=f"fixed_seeds[{index}]")
        object.__setattr__(
            self, "intended_train_patient_ids", tuple(self.intended_train_patient_ids)
        )
        object.__setattr__(
            self,
            "intended_validation_patient_ids",
            tuple(self.intended_validation_patient_ids),
        )
        _require_sorted_unique_ids(
            self.intended_train_patient_ids, field_name="intended_train_patient_ids"
        )
        _require_sorted_unique_ids(
            self.intended_validation_patient_ids,
            field_name="intended_validation_patient_ids",
        )
        if not self.intended_train_patient_ids or not self.intended_validation_patient_ids:
            raise Phase8RealDevelopmentValidationError(
                "run plan requires nonempty intended train and validation partitions."
            )
        _require_disjoint(
            self.intended_train_patient_ids,
            self.intended_validation_patient_ids,
            field_name="intended train/validation partitions",
        )
        _require_true(self.internal_test_excluded, field_name="internal_test_excluded")
        _require_identifier(
            self.expected_checkpoint_metadata_schema_name,
            field_name="expected_checkpoint_metadata_schema_name",
        )
        _require_schema_version(
            self.expected_checkpoint_metadata_schema_version,
            field_name="expected_checkpoint_metadata_schema_version",
        )
        _require_identifier(
            self.expected_validation_evidence_schema_name,
            field_name="expected_validation_evidence_schema_name",
        )
        _require_schema_version(
            self.expected_validation_evidence_schema_version,
            field_name="expected_validation_evidence_schema_version",
        )
        object.__setattr__(
            self,
            "expected_output_artifact_names",
            tuple(self.expected_output_artifact_names),
        )
        if not self.expected_output_artifact_names:
            raise Phase8RealDevelopmentValidationError(
                "expected_output_artifact_names must not be empty."
            )
        for index, name in enumerate(self.expected_output_artifact_names):
            _require_identifier(name, field_name=f"expected_output_artifact_names[{index}]")
        if len(set(self.expected_output_artifact_names)) != len(
            self.expected_output_artifact_names
        ):
            raise Phase8RealDevelopmentValidationError(
                "expected_output_artifact_names must not contain duplicates."
            )
        _require_true(self.no_external_data, field_name="no_external_data")
        _require_allowed(
            self.pixel_access_state,
            _ALLOWED_PIXEL_ACCESS_STATES,
            field_name="pixel_access_state",
        )
        if self.pixel_access_state != PIXEL_ACCESS_NOT_STARTED:
            raise Phase8RealDevelopmentValidationError(
                "Substage 2 run plans must use pixel_access_state='not_started'."
            )
        _require_allowed(
            self.execution_release_state,
            _ALLOWED_EXECUTION_RELEASE_STATES,
            field_name="execution_release_state",
        )
        if self.execution_release_state != EXECUTION_RELEASE_SCAFFOLD_ONLY:
            raise Phase8RealDevelopmentValidationError(
                "Substage 2 run plans must use execution_release_state='scaffold_only'."
            )
        _require_self_hash(
            self.run_plan_hash,
            _run_plan_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                input_binding_hash=self.input_binding_hash,
                fixed_candidate_inventory_hash=self.fixed_candidate_inventory_hash,
                candidate_id=self.candidate_id,
                model_family=self.model_family,
                training_adaptation_mode=self.training_adaptation_mode,
                training_config_reference=self.training_config_reference,
                preprocessing_decision_reference=self.preprocessing_decision_reference,
                fixed_seeds=self.fixed_seeds,
                intended_train_patient_ids=self.intended_train_patient_ids,
                intended_validation_patient_ids=self.intended_validation_patient_ids,
                internal_test_excluded=self.internal_test_excluded,
                expected_checkpoint_metadata_schema_name=(
                    self.expected_checkpoint_metadata_schema_name
                ),
                expected_checkpoint_metadata_schema_version=(
                    self.expected_checkpoint_metadata_schema_version
                ),
                expected_validation_evidence_schema_name=(
                    self.expected_validation_evidence_schema_name
                ),
                expected_validation_evidence_schema_version=(
                    self.expected_validation_evidence_schema_version
                ),
                expected_output_artifact_names=self.expected_output_artifact_names,
                no_external_data=self.no_external_data,
                pixel_access_state=self.pixel_access_state,
                execution_release_state=self.execution_release_state,
            ),
            field_name="run_plan_hash",
        )


def _run_plan_payload(
    *,
    schema_name: str,
    schema_version: str,
    input_binding_hash: str,
    fixed_candidate_inventory_hash: str,
    candidate_id: str,
    model_family: str,
    training_adaptation_mode: str,
    training_config_reference: ArtifactReference,
    preprocessing_decision_reference: ArtifactReference,
    fixed_seeds: Sequence[int],
    intended_train_patient_ids: Sequence[str],
    intended_validation_patient_ids: Sequence[str],
    internal_test_excluded: bool,
    expected_checkpoint_metadata_schema_name: str,
    expected_checkpoint_metadata_schema_version: str,
    expected_validation_evidence_schema_name: str,
    expected_validation_evidence_schema_version: str,
    expected_output_artifact_names: Sequence[str],
    no_external_data: bool,
    pixel_access_state: str,
    execution_release_state: str,
) -> dict[str, JsonValue]:
    return {
        "candidate_id": candidate_id,
        "execution_release_state": execution_release_state,
        "expected_checkpoint_metadata_schema_name": expected_checkpoint_metadata_schema_name,
        "expected_checkpoint_metadata_schema_version": (
            expected_checkpoint_metadata_schema_version
        ),
        "expected_output_artifact_names": list(expected_output_artifact_names),
        "expected_validation_evidence_schema_name": expected_validation_evidence_schema_name,
        "expected_validation_evidence_schema_version": (
            expected_validation_evidence_schema_version
        ),
        "fixed_candidate_inventory_hash": fixed_candidate_inventory_hash,
        "fixed_seeds": list(fixed_seeds),
        "input_binding_hash": input_binding_hash,
        "intended_train_patient_ids": list(intended_train_patient_ids),
        "intended_validation_patient_ids": list(intended_validation_patient_ids),
        "internal_test_excluded": internal_test_excluded,
        "model_family": model_family,
        "no_external_data": no_external_data,
        "pixel_access_state": pixel_access_state,
        "preprocessing_decision_reference": artifact_reference_to_dict(
            preprocessing_decision_reference
        ),
        "schema_name": schema_name,
        "schema_version": schema_version,
        "training_adaptation_mode": training_adaptation_mode,
        "training_config_reference": artifact_reference_to_dict(training_config_reference),
    }


def phase8_real_development_run_plan_identity_payload(
    plan: Phase8RealDevelopmentRunPlan,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for a run plan."""

    return _run_plan_payload(
        schema_name=plan.schema_name,
        schema_version=plan.schema_version,
        input_binding_hash=plan.input_binding_hash,
        fixed_candidate_inventory_hash=plan.fixed_candidate_inventory_hash,
        candidate_id=plan.candidate_id,
        model_family=plan.model_family,
        training_adaptation_mode=plan.training_adaptation_mode,
        training_config_reference=plan.training_config_reference,
        preprocessing_decision_reference=plan.preprocessing_decision_reference,
        fixed_seeds=plan.fixed_seeds,
        intended_train_patient_ids=plan.intended_train_patient_ids,
        intended_validation_patient_ids=plan.intended_validation_patient_ids,
        internal_test_excluded=plan.internal_test_excluded,
        expected_checkpoint_metadata_schema_name=plan.expected_checkpoint_metadata_schema_name,
        expected_checkpoint_metadata_schema_version=(
            plan.expected_checkpoint_metadata_schema_version
        ),
        expected_validation_evidence_schema_name=plan.expected_validation_evidence_schema_name,
        expected_validation_evidence_schema_version=(
            plan.expected_validation_evidence_schema_version
        ),
        expected_output_artifact_names=plan.expected_output_artifact_names,
        no_external_data=plan.no_external_data,
        pixel_access_state=plan.pixel_access_state,
        execution_release_state=plan.execution_release_state,
    )


def hash_phase8_real_development_run_plan(plan: Phase8RealDevelopmentRunPlan) -> str:
    """Return the canonical self-hash for a run plan."""

    return sha256_json(phase8_real_development_run_plan_identity_payload(plan))


def phase8_real_development_run_plan_to_dict(
    plan: Phase8RealDevelopmentRunPlan,
) -> dict[str, JsonValue]:
    """Convert a run plan to a canonical mapping."""

    payload = phase8_real_development_run_plan_identity_payload(plan)
    payload["run_plan_hash"] = plan.run_plan_hash
    return payload


def phase8_real_development_run_plan_to_json(plan: Phase8RealDevelopmentRunPlan) -> bytes:
    """Serialize a run plan to canonical JSON bytes."""

    return canonical_json_bytes(phase8_real_development_run_plan_to_dict(plan)) + b"\n"


_RUN_PLAN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "input_binding_hash",
        "fixed_candidate_inventory_hash",
        "candidate_id",
        "model_family",
        "training_adaptation_mode",
        "training_config_reference",
        "preprocessing_decision_reference",
        "fixed_seeds",
        "intended_train_patient_ids",
        "intended_validation_patient_ids",
        "internal_test_excluded",
        "expected_checkpoint_metadata_schema_name",
        "expected_checkpoint_metadata_schema_version",
        "expected_validation_evidence_schema_name",
        "expected_validation_evidence_schema_version",
        "expected_output_artifact_names",
        "no_external_data",
        "pixel_access_state",
        "execution_release_state",
        "run_plan_hash",
    }
)


def phase8_real_development_run_plan_from_mapping(
    mapping: MappingLike,
) -> Phase8RealDevelopmentRunPlan:
    """Reconstruct a run plan from a strict mapping."""

    _require_exact_fields(mapping, _RUN_PLAN_FIELDS, object_name="Phase8RealDevelopmentRunPlan")
    return Phase8RealDevelopmentRunPlan(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        input_binding_hash=_expect_string(
            mapping["input_binding_hash"], field_name="input_binding_hash"
        ),
        fixed_candidate_inventory_hash=_expect_string(
            mapping["fixed_candidate_inventory_hash"],
            field_name="fixed_candidate_inventory_hash",
        ),
        candidate_id=_expect_string(mapping["candidate_id"], field_name="candidate_id"),
        model_family=_expect_string(mapping["model_family"], field_name="model_family"),
        training_adaptation_mode=_expect_string(
            mapping["training_adaptation_mode"], field_name="training_adaptation_mode"
        ),
        training_config_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["training_config_reference"], field_name="training_config_reference"
            )
        ),
        preprocessing_decision_reference=artifact_reference_from_mapping(
            _expect_mapping(
                mapping["preprocessing_decision_reference"],
                field_name="preprocessing_decision_reference",
            )
        ),
        fixed_seeds=_expect_int_tuple(mapping["fixed_seeds"], field_name="fixed_seeds"),
        intended_train_patient_ids=_expect_string_tuple(
            mapping["intended_train_patient_ids"], field_name="intended_train_patient_ids"
        ),
        intended_validation_patient_ids=_expect_string_tuple(
            mapping["intended_validation_patient_ids"],
            field_name="intended_validation_patient_ids",
        ),
        internal_test_excluded=_expect_bool(
            mapping["internal_test_excluded"], field_name="internal_test_excluded"
        ),
        expected_checkpoint_metadata_schema_name=_expect_string(
            mapping["expected_checkpoint_metadata_schema_name"],
            field_name="expected_checkpoint_metadata_schema_name",
        ),
        expected_checkpoint_metadata_schema_version=_expect_string(
            mapping["expected_checkpoint_metadata_schema_version"],
            field_name="expected_checkpoint_metadata_schema_version",
        ),
        expected_validation_evidence_schema_name=_expect_string(
            mapping["expected_validation_evidence_schema_name"],
            field_name="expected_validation_evidence_schema_name",
        ),
        expected_validation_evidence_schema_version=_expect_string(
            mapping["expected_validation_evidence_schema_version"],
            field_name="expected_validation_evidence_schema_version",
        ),
        expected_output_artifact_names=_expect_string_tuple(
            mapping["expected_output_artifact_names"],
            field_name="expected_output_artifact_names",
        ),
        no_external_data=_expect_bool(mapping["no_external_data"], field_name="no_external_data"),
        pixel_access_state=_expect_string(
            mapping["pixel_access_state"], field_name="pixel_access_state"
        ),
        execution_release_state=_expect_string(
            mapping["execution_release_state"], field_name="execution_release_state"
        ),
        run_plan_hash=_expect_string(mapping["run_plan_hash"], field_name="run_plan_hash"),
    )


def build_phase8_real_development_run_plan(
    *,
    input_binding: Phase8RealDevelopmentInputBinding,
    candidate_inventory: Phase8FixedCandidateInventory,
    candidate_id: str,
    training_config_reference: ArtifactReference,
    preprocessing_decision_reference: ArtifactReference,
    fixed_seeds: tuple[int, ...],
    expected_checkpoint_metadata_schema: tuple[str, str],
    expected_validation_evidence_schema: tuple[str, str],
    expected_output_artifact_names: tuple[str, ...],
) -> Phase8RealDevelopmentRunPlan:
    """Build a scaffold-only run plan bound to one fixed inventory candidate.

    Always produces ``pixel_access_state='not_started'`` and
    ``execution_release_state='scaffold_only'``; there is no parameter that can
    change either value.
    """

    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_inventory.candidates
    }
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        raise Phase8RealDevelopmentValidationError(
            f"candidate_id {candidate_id!r} is not present in the fixed candidate inventory."
        )
    if training_config_reference.artifact_hash != candidate.training_config_reference.artifact_hash:
        raise Phase8RealDevelopmentValidationError(
            "training_config_reference does not match the candidate's fixed training config."
        )
    if preprocessing_decision_reference.artifact_hash != candidate.preprocessing_evidence_hash:
        raise Phase8RealDevelopmentValidationError(
            "preprocessing_decision_reference does not match the candidate's fixed "
            "preprocessing evidence."
        )
    if preprocessing_decision_reference.artifact_hash == _ZERO_SHA256:
        raise Phase8RealDevelopmentValidationError(
            "preprocessing_decision_reference must not use a placeholder hash."
        )

    intended_train_patient_ids = input_binding.train_patient_ids
    intended_validation_patient_ids = input_binding.validation_patient_ids
    internal_test_patient_ids = set(input_binding.internal_test_patient_ids)
    if internal_test_patient_ids.intersection(
        intended_train_patient_ids
    ) or internal_test_patient_ids.intersection(intended_validation_patient_ids):
        raise Phase8RealDevelopmentValidationError(
            "internal_test patients must not appear in intended train/validation partitions."
        )

    checkpoint_schema_name, checkpoint_schema_version = expected_checkpoint_metadata_schema
    validation_schema_name, validation_schema_version = expected_validation_evidence_schema
    fixed_seeds_tuple = tuple(fixed_seeds)
    expected_output_artifact_names_tuple = tuple(expected_output_artifact_names)

    payload = _run_plan_payload(
        schema_name=PHASE8_REAL_DEVELOPMENT_RUN_PLAN_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        input_binding_hash=input_binding.input_binding_hash,
        fixed_candidate_inventory_hash=candidate_inventory.inventory_hash,
        candidate_id=candidate_id,
        model_family=candidate.model_family,
        training_adaptation_mode=candidate.training_adaptation_mode,
        training_config_reference=training_config_reference,
        preprocessing_decision_reference=preprocessing_decision_reference,
        fixed_seeds=fixed_seeds_tuple,
        intended_train_patient_ids=intended_train_patient_ids,
        intended_validation_patient_ids=intended_validation_patient_ids,
        internal_test_excluded=True,
        expected_checkpoint_metadata_schema_name=checkpoint_schema_name,
        expected_checkpoint_metadata_schema_version=checkpoint_schema_version,
        expected_validation_evidence_schema_name=validation_schema_name,
        expected_validation_evidence_schema_version=validation_schema_version,
        expected_output_artifact_names=expected_output_artifact_names_tuple,
        no_external_data=True,
        pixel_access_state=PIXEL_ACCESS_NOT_STARTED,
        execution_release_state=EXECUTION_RELEASE_SCAFFOLD_ONLY,
    )
    return Phase8RealDevelopmentRunPlan(
        schema_name=PHASE8_REAL_DEVELOPMENT_RUN_PLAN_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        input_binding_hash=input_binding.input_binding_hash,
        fixed_candidate_inventory_hash=candidate_inventory.inventory_hash,
        candidate_id=candidate_id,
        model_family=candidate.model_family,
        training_adaptation_mode=candidate.training_adaptation_mode,
        training_config_reference=training_config_reference,
        preprocessing_decision_reference=preprocessing_decision_reference,
        fixed_seeds=fixed_seeds_tuple,
        intended_train_patient_ids=intended_train_patient_ids,
        intended_validation_patient_ids=intended_validation_patient_ids,
        internal_test_excluded=True,
        expected_checkpoint_metadata_schema_name=checkpoint_schema_name,
        expected_checkpoint_metadata_schema_version=checkpoint_schema_version,
        expected_validation_evidence_schema_name=validation_schema_name,
        expected_validation_evidence_schema_version=validation_schema_version,
        expected_output_artifact_names=expected_output_artifact_names_tuple,
        no_external_data=True,
        pixel_access_state=PIXEL_ACCESS_NOT_STARTED,
        execution_release_state=EXECUTION_RELEASE_SCAFFOLD_ONLY,
        run_plan_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# C. Case-binding plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8RealDevelopmentCaseBinding:
    """Per-case metadata-only binding; never reads image or label bytes."""

    schema_name: str
    schema_version: str
    anonymous_patient_id: str
    anonymous_case_id: str
    partition: str
    manifest_record_identity_hash: str
    relative_image_path: str
    relative_label_path: str
    pixel_access_state: str
    case_binding_hash: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_name, PHASE8_REAL_DEVELOPMENT_CASE_BINDING_SCHEMA_NAME)
        _require_schema_version_exact(self.schema_version)
        _require_identifier(self.anonymous_patient_id, field_name="anonymous_patient_id")
        _require_identifier(self.anonymous_case_id, field_name="anonymous_case_id")
        _require_allowed(self.partition, _ALLOWED_CASE_BINDING_PARTITIONS, field_name="partition")
        _require_sha256(
            self.manifest_record_identity_hash, field_name="manifest_record_identity_hash"
        )
        if normalize_safe_relative_posix_path(self.relative_image_path) != (
            self.relative_image_path
        ):
            raise Phase8RealDevelopmentValidationError(
                "relative_image_path must already be normalized."
            )
        if normalize_safe_relative_posix_path(self.relative_label_path) != (
            self.relative_label_path
        ):
            raise Phase8RealDevelopmentValidationError(
                "relative_label_path must already be normalized."
            )
        if self.relative_image_path == self.relative_label_path:
            raise Phase8RealDevelopmentValidationError(
                "relative_image_path and relative_label_path must differ."
            )
        if self.pixel_access_state != PIXEL_ACCESS_NOT_STARTED:
            raise Phase8RealDevelopmentValidationError(
                "case bindings must use pixel_access_state='not_started'."
            )
        _require_self_hash(
            self.case_binding_hash,
            _case_binding_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                anonymous_patient_id=self.anonymous_patient_id,
                anonymous_case_id=self.anonymous_case_id,
                partition=self.partition,
                manifest_record_identity_hash=self.manifest_record_identity_hash,
                relative_image_path=self.relative_image_path,
                relative_label_path=self.relative_label_path,
                pixel_access_state=self.pixel_access_state,
            ),
            field_name="case_binding_hash",
        )


def _case_binding_payload(
    *,
    schema_name: str,
    schema_version: str,
    anonymous_patient_id: str,
    anonymous_case_id: str,
    partition: str,
    manifest_record_identity_hash: str,
    relative_image_path: str,
    relative_label_path: str,
    pixel_access_state: str,
) -> dict[str, JsonValue]:
    return {
        "anonymous_case_id": anonymous_case_id,
        "anonymous_patient_id": anonymous_patient_id,
        "manifest_record_identity_hash": manifest_record_identity_hash,
        "partition": partition,
        "pixel_access_state": pixel_access_state,
        "relative_image_path": relative_image_path,
        "relative_label_path": relative_label_path,
        "schema_name": schema_name,
        "schema_version": schema_version,
    }


def phase8_real_development_case_binding_identity_payload(
    binding: Phase8RealDevelopmentCaseBinding,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for one case binding."""

    return _case_binding_payload(
        schema_name=binding.schema_name,
        schema_version=binding.schema_version,
        anonymous_patient_id=binding.anonymous_patient_id,
        anonymous_case_id=binding.anonymous_case_id,
        partition=binding.partition,
        manifest_record_identity_hash=binding.manifest_record_identity_hash,
        relative_image_path=binding.relative_image_path,
        relative_label_path=binding.relative_label_path,
        pixel_access_state=binding.pixel_access_state,
    )


def hash_phase8_real_development_case_binding(
    binding: Phase8RealDevelopmentCaseBinding,
) -> str:
    """Return the canonical self-hash for one case binding."""

    return sha256_json(phase8_real_development_case_binding_identity_payload(binding))


def phase8_real_development_case_binding_to_dict(
    binding: Phase8RealDevelopmentCaseBinding,
) -> dict[str, JsonValue]:
    """Convert one case binding to a canonical mapping."""

    payload = phase8_real_development_case_binding_identity_payload(binding)
    payload["case_binding_hash"] = binding.case_binding_hash
    return payload


_CASE_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "anonymous_patient_id",
        "anonymous_case_id",
        "partition",
        "manifest_record_identity_hash",
        "relative_image_path",
        "relative_label_path",
        "pixel_access_state",
        "case_binding_hash",
    }
)


def phase8_real_development_case_binding_from_mapping(
    mapping: MappingLike,
) -> Phase8RealDevelopmentCaseBinding:
    """Reconstruct one case binding from a strict mapping."""

    _require_exact_fields(
        mapping, _CASE_BINDING_FIELDS, object_name="Phase8RealDevelopmentCaseBinding"
    )
    return Phase8RealDevelopmentCaseBinding(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        anonymous_patient_id=_expect_string(
            mapping["anonymous_patient_id"], field_name="anonymous_patient_id"
        ),
        anonymous_case_id=_expect_string(
            mapping["anonymous_case_id"], field_name="anonymous_case_id"
        ),
        partition=_expect_string(mapping["partition"], field_name="partition"),
        manifest_record_identity_hash=_expect_string(
            mapping["manifest_record_identity_hash"],
            field_name="manifest_record_identity_hash",
        ),
        relative_image_path=_expect_string(
            mapping["relative_image_path"], field_name="relative_image_path"
        ),
        relative_label_path=_expect_string(
            mapping["relative_label_path"], field_name="relative_label_path"
        ),
        pixel_access_state=_expect_string(
            mapping["pixel_access_state"], field_name="pixel_access_state"
        ),
        case_binding_hash=_expect_string(
            mapping["case_binding_hash"], field_name="case_binding_hash"
        ),
    )


@dataclass(frozen=True, slots=True)
class Phase8RealDevelopmentCaseBindingCollection:
    """Deterministically ordered collection of per-case bindings for one run plan."""

    schema_name: str
    schema_version: str
    run_plan_hash: str
    bindings: tuple[Phase8RealDevelopmentCaseBinding, ...]
    train_count: int
    validation_count: int
    collection_hash: str

    def __post_init__(self) -> None:
        _require_schema(
            self.schema_name, PHASE8_REAL_DEVELOPMENT_CASE_BINDING_COLLECTION_SCHEMA_NAME
        )
        _require_schema_version_exact(self.schema_version)
        _require_sha256(self.run_plan_hash, field_name="run_plan_hash")
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if not self.bindings:
            raise Phase8RealDevelopmentValidationError("case binding collection must not be empty.")
        ordered = tuple(
            sorted(
                self.bindings,
                key=lambda item: (
                    item.partition,
                    item.anonymous_patient_id,
                    item.anonymous_case_id,
                ),
            )
        )
        if ordered != self.bindings:
            raise Phase8RealDevelopmentValidationError(
                "case bindings must be sorted by (partition, patient_id, case_id)."
            )
        case_ids = [item.anonymous_case_id for item in self.bindings]
        if len(case_ids) != len(set(case_ids)):
            raise Phase8RealDevelopmentValidationError(
                "case bindings must not contain duplicate anonymous_case_id values."
            )
        actual_train_count = sum(1 for item in self.bindings if item.partition == "train")
        actual_validation_count = sum(1 for item in self.bindings if item.partition == "validation")
        if actual_train_count != self.train_count:
            raise Phase8RealDevelopmentValidationError("train_count does not match bindings.")
        if actual_validation_count != self.validation_count:
            raise Phase8RealDevelopmentValidationError("validation_count does not match bindings.")
        _require_self_hash(
            self.collection_hash,
            _case_binding_collection_payload(
                schema_name=self.schema_name,
                schema_version=self.schema_version,
                run_plan_hash=self.run_plan_hash,
                bindings=self.bindings,
                train_count=self.train_count,
                validation_count=self.validation_count,
            ),
            field_name="collection_hash",
        )


def _case_binding_collection_payload(
    *,
    schema_name: str,
    schema_version: str,
    run_plan_hash: str,
    bindings: Sequence[Phase8RealDevelopmentCaseBinding],
    train_count: int,
    validation_count: int,
) -> dict[str, JsonValue]:
    return {
        "bindings": [phase8_real_development_case_binding_to_dict(item) for item in bindings],
        "run_plan_hash": run_plan_hash,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "train_count": train_count,
        "validation_count": validation_count,
    }


def phase8_real_development_case_binding_collection_identity_payload(
    collection: Phase8RealDevelopmentCaseBindingCollection,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload for a case binding collection."""

    return _case_binding_collection_payload(
        schema_name=collection.schema_name,
        schema_version=collection.schema_version,
        run_plan_hash=collection.run_plan_hash,
        bindings=collection.bindings,
        train_count=collection.train_count,
        validation_count=collection.validation_count,
    )


def hash_phase8_real_development_case_binding_collection(
    collection: Phase8RealDevelopmentCaseBindingCollection,
) -> str:
    """Return the canonical self-hash for a case binding collection."""

    return sha256_json(phase8_real_development_case_binding_collection_identity_payload(collection))


def phase8_real_development_case_binding_collection_to_dict(
    collection: Phase8RealDevelopmentCaseBindingCollection,
) -> dict[str, JsonValue]:
    """Convert a case binding collection to a canonical mapping."""

    payload = phase8_real_development_case_binding_collection_identity_payload(collection)
    payload["collection_hash"] = collection.collection_hash
    return payload


def phase8_real_development_case_binding_collection_to_json(
    collection: Phase8RealDevelopmentCaseBindingCollection,
) -> bytes:
    """Serialize a case binding collection to canonical JSON bytes."""

    return (
        canonical_json_bytes(phase8_real_development_case_binding_collection_to_dict(collection))
        + b"\n"
    )


_CASE_BINDING_COLLECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_name",
        "schema_version",
        "run_plan_hash",
        "bindings",
        "train_count",
        "validation_count",
        "collection_hash",
    }
)


def phase8_real_development_case_binding_collection_from_mapping(
    mapping: MappingLike,
) -> Phase8RealDevelopmentCaseBindingCollection:
    """Reconstruct a case binding collection from a strict mapping."""

    _require_exact_fields(
        mapping,
        _CASE_BINDING_COLLECTION_FIELDS,
        object_name="Phase8RealDevelopmentCaseBindingCollection",
    )
    bindings_value = _expect_list(mapping["bindings"], field_name="bindings")
    bindings = tuple(
        phase8_real_development_case_binding_from_mapping(
            _expect_mapping(item, field_name=f"bindings[{index}]")
        )
        for index, item in enumerate(bindings_value)
    )
    return Phase8RealDevelopmentCaseBindingCollection(
        schema_name=_expect_string(mapping["schema_name"], field_name="schema_name"),
        schema_version=_expect_string(mapping["schema_version"], field_name="schema_version"),
        run_plan_hash=_expect_string(mapping["run_plan_hash"], field_name="run_plan_hash"),
        bindings=bindings,
        train_count=_expect_int(mapping["train_count"], field_name="train_count"),
        validation_count=_expect_int(mapping["validation_count"], field_name="validation_count"),
        collection_hash=_expect_string(mapping["collection_hash"], field_name="collection_hash"),
    )


def _manifest_record_identity_hash(
    *,
    anonymous_patient_id: str,
    anonymous_case_id: str,
    image_sha256: str,
    label_sha256: str,
    relative_image_path: str,
    relative_label_path: str,
) -> str:
    return sha256_json(
        {
            "anonymous_case_id": anonymous_case_id,
            "anonymous_patient_id": anonymous_patient_id,
            "image_sha256": image_sha256,
            "label_sha256": label_sha256,
            "relative_image_path": relative_image_path,
            "relative_label_path": relative_label_path,
        }
    )


def build_phase8_real_development_case_bindings(
    *,
    run_plan: Phase8RealDevelopmentRunPlan,
    manifest: DatasetManifest,
) -> tuple[Phase8RealDevelopmentCaseBinding, ...]:
    """Build train/validation case bindings from an already-verified manifest.

    Never opens image or label files; every field is derived from the
    already-parsed :class:`DatasetManifest` records and the run plan's
    intended patient-ID partitions. Cases outside those two partitions
    (including any internal-test case) are silently excluded, and no
    ``internal_test`` binding is ever produced.
    """

    train_ids = set(run_plan.intended_train_patient_ids)
    validation_ids = set(run_plan.intended_validation_patient_ids)
    bindings: list[Phase8RealDevelopmentCaseBinding] = []
    seen_case_ids: set[str] = set()
    for case in manifest.cases:
        if case.anonymous_patient_id in train_ids:
            partition = "train"
        elif case.anonymous_patient_id in validation_ids:
            partition = "validation"
        else:
            continue
        if case.anonymous_case_id in seen_case_ids:
            raise Phase8RealDevelopmentValidationError(
                f"duplicate anonymous_case_id in manifest: {case.anonymous_case_id!r}"
            )
        seen_case_ids.add(case.anonymous_case_id)

        relative_image_path = normalize_safe_relative_posix_path(case.relative_image_path)
        relative_label_path = normalize_safe_relative_posix_path(case.relative_label_path)
        identity_hash = _manifest_record_identity_hash(
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            image_sha256=case.image_sha256,
            label_sha256=case.label_sha256,
            relative_image_path=relative_image_path,
            relative_label_path=relative_label_path,
        )
        payload = _case_binding_payload(
            schema_name=PHASE8_REAL_DEVELOPMENT_CASE_BINDING_SCHEMA_NAME,
            schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
            anonymous_patient_id=case.anonymous_patient_id,
            anonymous_case_id=case.anonymous_case_id,
            partition=partition,
            manifest_record_identity_hash=identity_hash,
            relative_image_path=relative_image_path,
            relative_label_path=relative_label_path,
            pixel_access_state=PIXEL_ACCESS_NOT_STARTED,
        )
        bindings.append(
            Phase8RealDevelopmentCaseBinding(
                schema_name=PHASE8_REAL_DEVELOPMENT_CASE_BINDING_SCHEMA_NAME,
                schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
                anonymous_patient_id=case.anonymous_patient_id,
                anonymous_case_id=case.anonymous_case_id,
                partition=partition,
                manifest_record_identity_hash=identity_hash,
                relative_image_path=relative_image_path,
                relative_label_path=relative_label_path,
                pixel_access_state=PIXEL_ACCESS_NOT_STARTED,
                case_binding_hash=sha256_json(payload),
            )
        )
    if not bindings:
        raise Phase8RealDevelopmentValidationError(
            "case-binding builder produced no bindings for the run plan partitions."
        )
    return tuple(
        sorted(
            bindings,
            key=lambda item: (item.partition, item.anonymous_patient_id, item.anonymous_case_id),
        )
    )


def build_phase8_real_development_case_binding_collection(
    *,
    run_plan: Phase8RealDevelopmentRunPlan,
    case_bindings: tuple[Phase8RealDevelopmentCaseBinding, ...],
) -> Phase8RealDevelopmentCaseBindingCollection:
    """Build a deterministically ordered case-binding collection for one run plan."""

    ordered = tuple(
        sorted(
            case_bindings,
            key=lambda item: (item.partition, item.anonymous_patient_id, item.anonymous_case_id),
        )
    )
    train_count = sum(1 for item in ordered if item.partition == "train")
    validation_count = sum(1 for item in ordered if item.partition == "validation")
    payload = _case_binding_collection_payload(
        schema_name=PHASE8_REAL_DEVELOPMENT_CASE_BINDING_COLLECTION_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        run_plan_hash=run_plan.run_plan_hash,
        bindings=ordered,
        train_count=train_count,
        validation_count=validation_count,
    )
    return Phase8RealDevelopmentCaseBindingCollection(
        schema_name=PHASE8_REAL_DEVELOPMENT_CASE_BINDING_COLLECTION_SCHEMA_NAME,
        schema_version=PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        run_plan_hash=run_plan.run_plan_hash,
        bindings=ordered,
        train_count=train_count,
        validation_count=validation_count,
        collection_hash=sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# D. Execution boundary interface (Substage 3 stub only; not implemented here)
# ---------------------------------------------------------------------------


class Phase8RealDevelopmentExecutor(Protocol):
    """Method surface a future Substage 3 executor must implement.

    Every method here is a typed stub only. This module never calls, imports
    an implementation of, or otherwise exercises any of these methods.
    """

    def construct_model(self, plan: Phase8RealDevelopmentRunPlan) -> object:
        """Construct the model architecture described by the run plan."""
        raise NotImplementedError

    def load_training_data(
        self,
        plan: Phase8RealDevelopmentRunPlan,
        case_bindings: tuple[Phase8RealDevelopmentCaseBinding, ...],
    ) -> object:
        """Load the train-partition data referenced by the case bindings."""
        raise NotImplementedError

    def run_validation_inference(
        self,
        plan: Phase8RealDevelopmentRunPlan,
        case_bindings: tuple[Phase8RealDevelopmentCaseBinding, ...],
    ) -> object:
        """Run validation-partition inference and return raw predictions."""
        raise NotImplementedError

    def persist_checkpoint(self, plan: Phase8RealDevelopmentRunPlan, model: object) -> object:
        """Persist a trained checkpoint and return its identity metadata."""
        raise NotImplementedError

    def publish_validation_metrics(
        self, plan: Phase8RealDevelopmentRunPlan, predictions: object
    ) -> object:
        """Compute and publish development-validation metrics evidence."""
        raise NotImplementedError

    def publish_preprocessing_evidence(self, plan: Phase8RealDevelopmentRunPlan) -> object:
        """Publish the preprocessing evidence actually applied during execution."""
        raise NotImplementedError

    def publish_checkpoint_metadata(
        self, plan: Phase8RealDevelopmentRunPlan, checkpoint: object
    ) -> object:
        """Publish checkpoint metadata linking the checkpoint to this run plan."""
        raise NotImplementedError


def execute_phase8_real_development_run(
    plan: Phase8RealDevelopmentRunPlan,
    executor: Phase8RealDevelopmentExecutor | None = None,
    **kwargs: object,
) -> None:
    """Fail closed: Substage 3 real-development execution is never released here.

    This function unconditionally raises
    :class:`Phase8RealDevelopmentExecutionNotReleasedError` regardless of the
    supplied plan, executor, or any keyword arguments. No caller-supplied value
    can bypass this guard.
    """

    raise Phase8RealDevelopmentExecutionNotReleasedError(
        "execution not released: Substage 2 scaffold only; Substage 3 requires "
        "explicit user approval."
    )


# ---------------------------------------------------------------------------
# E. Output-root safety and guarded scaffold-plan publication
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase8RealDevelopmentPlanPublicationResult:
    """Result of one guarded Phase 8 real-development scaffold-plan publication."""

    output_root: Path
    input_binding_hash: str
    run_plan_hash: str
    case_binding_collection_hash: str
    summary_hash: str
    artifact_hashes: dict[str, str]


def _validate_scaffold_output_root(*, output_root: Path, repository_root: Path) -> Path:
    if output_root.exists() and output_root.is_symlink():
        raise Phase8RealDevelopmentPublicationError("output_root must not be a symlink.")
    try:
        return validate_explicit_external_output_root(
            output_root, forbidden_roots=(repository_root,)
        )
    except InvalidDatasetRootError as exc:
        raise Phase8RealDevelopmentPublicationError("invalid scaffold output root.") from exc


def _require_output_root_empty_or_absent(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise Phase8RealDevelopmentPublicationError("scaffold output root already contains files.")


def _publish_json_bytes(path: Path, data: bytes) -> None:
    try:
        publish_text_no_overwrite(
            text=data.decode("utf-8"),
            output_path=path,
            temporary_exists_message=(
                "temporary Phase 8 real-development scaffold output already exists"
            ),
            final_exists_message="Phase 8 real-development scaffold output already exists",
        )
    except (Phase2PublicationExistingOutputError, Phase2PublicationIOError) as exc:
        raise Phase8RealDevelopmentPublicationError("failed to publish scaffold artifact.") from exc


def run_phase8_real_development_plan_publication(
    *,
    input_binding: Phase8RealDevelopmentInputBinding,
    run_plan: Phase8RealDevelopmentRunPlan,
    case_bindings: tuple[Phase8RealDevelopmentCaseBinding, ...],
    output_root: Path,
    repository_root: Path,
) -> Phase8RealDevelopmentPlanPublicationResult:
    """Publish the input binding, run plan, case bindings, and plan summary.

    Deterministic and byte-identical across two separate empty output roots
    given identical inputs; no timestamps or local paths appear in any
    published payload.
    """

    if run_plan.input_binding_hash != input_binding.input_binding_hash:
        raise Phase8RealDevelopmentPublicationError(
            "run plan does not reference the supplied input binding."
        )
    canonical_repository_root = repository_root.resolve(strict=True)
    canonical_output_root = _validate_scaffold_output_root(
        output_root=output_root, repository_root=canonical_repository_root
    )
    _require_output_root_empty_or_absent(canonical_output_root)

    collection = build_phase8_real_development_case_binding_collection(
        run_plan=run_plan, case_bindings=case_bindings
    )

    _publish_json_bytes(
        canonical_output_root / PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_FILENAME,
        phase8_real_development_input_binding_to_json(input_binding),
    )
    _publish_json_bytes(
        canonical_output_root / PHASE8_REAL_DEVELOPMENT_RUN_PLAN_FILENAME,
        phase8_real_development_run_plan_to_json(run_plan),
    )
    _publish_json_bytes(
        canonical_output_root / PHASE8_REAL_DEVELOPMENT_CASE_BINDINGS_FILENAME,
        phase8_real_development_case_binding_collection_to_json(collection),
    )

    summary_payload: dict[str, JsonValue] = {
        "case_binding_collection_hash": collection.collection_hash,
        "checkpoint_created": False,
        "execution_release_state": run_plan.execution_release_state,
        "input_binding_hash": input_binding.input_binding_hash,
        "pixel_access_not_started": run_plan.pixel_access_state == PIXEL_ACCESS_NOT_STARTED,
        "pixel_access_state": run_plan.pixel_access_state,
        "real_metrics_computed": False,
        "run_plan_hash": run_plan.run_plan_hash,
        "scaffold_only": run_plan.execution_release_state == EXECUTION_RELEASE_SCAFFOLD_ONLY,
        "schema_name": PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_SCHEMA_NAME,
        "schema_version": PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION,
        "train_case_count": collection.train_count,
        "training_executed": False,
        "validation_case_count": collection.validation_count,
    }
    summary_payload["summary_identity_hash"] = sha256_json(summary_payload)
    _publish_json_bytes(
        canonical_output_root / PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME,
        canonical_json_bytes(summary_payload) + b"\n",
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(canonical_output_root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    return Phase8RealDevelopmentPlanPublicationResult(
        output_root=canonical_output_root,
        input_binding_hash=input_binding.input_binding_hash,
        run_plan_hash=run_plan.run_plan_hash,
        case_binding_collection_hash=collection.collection_hash,
        summary_hash=artifact_hashes[PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME],
        artifact_hashes=artifact_hashes,
    )


# ---------------------------------------------------------------------------
# Shared validation and serialization helpers
# ---------------------------------------------------------------------------


def _require_schema(value: str, expected: str) -> None:
    if value != expected:
        raise Phase8RealDevelopmentValidationError(
            f"schema_name must equal {expected!r}, got {value!r}."
        )


def _require_schema_version_exact(value: str) -> None:
    if value != PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION:
        raise Phase8RealDevelopmentValidationError(
            f"schema_version must equal {PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION!r}."
        )


def _require_schema_version(value: str, *, field_name: str) -> None:
    if not re.fullmatch(r"^v[1-9][0-9]*$", value):
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be a vN schema version.")


def _require_allowed(value: str, allowed: frozenset[str], *, field_name: str) -> None:
    if value not in allowed:
        raise Phase8RealDevelopmentValidationError(
            f"{field_name} must be one of {sorted(allowed)!r}, got {value!r}."
        )


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise Phase8RealDevelopmentValidationError(
            f"{field_name} must be a lowercase safe identifier."
        )
    if value in _UNRESOLVED_TOKENS or _contains_unresolved_token(value):
        raise Phase8RealDevelopmentValidationError(f"{field_name} must not be unresolved.")
    _require_no_absolute_path(value, field_name=field_name)


def _require_nonempty_token(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or value == "":
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be a nonempty string.")
    if value != value.strip():
        raise Phase8RealDevelopmentValidationError(
            f"{field_name} must not contain leading or trailing whitespace."
        )
    _require_no_absolute_path(value, field_name=field_name)


def _require_no_absolute_path(value: str, *, field_name: str) -> None:
    if value.startswith("/") or value.startswith("~") or "\\" in value:
        raise Phase8RealDevelopmentValidationError(
            f"{field_name} must not contain local path identity."
        )
    if re.match(r"^[A-Za-z]:[\\/]", value):
        raise Phase8RealDevelopmentValidationError(
            f"{field_name} must not contain an absolute path."
        )
    if "://" in value:
        raise Phase8RealDevelopmentValidationError(f"{field_name} must not contain URI identity.")


def _contains_unresolved_token(value: str) -> bool:
    return any(part in _UNRESOLVED_TOKENS for part in re.split(r"[_.@:/-]+", value))


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise Phase8RealDevelopmentHashError(f"{field_name} must be a lowercase SHA-256 digest.")
    if value == _ZERO_SHA256:
        raise Phase8RealDevelopmentHashError(f"{field_name} must not be the zero SHA-256 digest.")


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be a nonnegative integer.")


def _require_true(value: bool, *, field_name: str) -> None:
    if value is not True:
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be true.")


def _require_sorted_unique_ids(values: tuple[str, ...], *, field_name: str) -> None:
    if tuple(sorted(values)) != values:
        raise Phase8RealDevelopmentValidationError(f"{field_name} must be sorted.")
    if len(set(values)) != len(values):
        raise Phase8RealDevelopmentValidationError(f"{field_name} must not contain duplicates.")
    for index, value in enumerate(values):
        _require_identifier(value, field_name=f"{field_name}[{index}]")


def _require_disjoint(*groups: Sequence[str], field_name: str) -> None:
    seen: set[str] = set()
    for group in groups:
        group_set = set(group)
        if seen.intersection(group_set):
            raise Phase8RealDevelopmentValidationError(f"{field_name} must be disjoint.")
        seen |= group_set


def _require_self_hash(value: str, payload: dict[str, JsonValue], *, field_name: str) -> None:
    _require_sha256(value, field_name=field_name)
    expected = sha256_json(payload)
    if value != expected:
        raise Phase8RealDevelopmentHashError(f"{field_name} does not match canonical identity.")


def _require_exact_fields(
    mapping: MappingLike, required_fields: frozenset[str], *, object_name: str
) -> None:
    keys = set(mapping)
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)
    if missing or extra:
        raise Phase8RealDevelopmentSerializationError(
            f"{object_name} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _expect_mapping(value: object, *, field_name: str) -> MappingLike:
    if not isinstance(value, Mapping):
        raise Phase8RealDevelopmentSerializationError(f"{field_name} must be a mapping.")
    return cast(MappingLike, value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise Phase8RealDevelopmentSerializationError(f"{field_name} must be a string.")
    return value


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise Phase8RealDevelopmentSerializationError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Phase8RealDevelopmentSerializationError(f"{field_name} must be an integer.")
    return value


def _expect_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise Phase8RealDevelopmentSerializationError(f"{field_name} must be a list.")
    return value


def _expect_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    sequence = _expect_list(value, field_name=field_name)
    return tuple(
        _expect_string(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(sequence)
    )


def _expect_int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    sequence = _expect_list(value, field_name=field_name)
    return tuple(
        _expect_int(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(sequence)
    )


__all__ = [
    "EXECUTION_RELEASE_SCAFFOLD_ONLY",
    "PHASE8_REAL_DEVELOPMENT_CASE_BINDINGS_FILENAME",
    "PHASE8_REAL_DEVELOPMENT_CASE_BINDING_COLLECTION_SCHEMA_NAME",
    "PHASE8_REAL_DEVELOPMENT_CASE_BINDING_SCHEMA_NAME",
    "PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_FILENAME",
    "PHASE8_REAL_DEVELOPMENT_INPUT_BINDING_SCHEMA_NAME",
    "PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_FILENAME",
    "PHASE8_REAL_DEVELOPMENT_PLAN_SUMMARY_SCHEMA_NAME",
    "PHASE8_REAL_DEVELOPMENT_RUN_PLAN_FILENAME",
    "PHASE8_REAL_DEVELOPMENT_RUN_PLAN_SCHEMA_NAME",
    "PHASE8_REAL_DEVELOPMENT_SCHEMA_VERSION",
    "PIXEL_ACCESS_NOT_STARTED",
    "Phase8RealDevelopmentCaseBinding",
    "Phase8RealDevelopmentCaseBindingCollection",
    "Phase8RealDevelopmentExecutionNotReleasedError",
    "Phase8RealDevelopmentExecutor",
    "Phase8RealDevelopmentHashError",
    "Phase8RealDevelopmentInputBinding",
    "Phase8RealDevelopmentPlanPublicationResult",
    "Phase8RealDevelopmentPublicationError",
    "Phase8RealDevelopmentRunPlan",
    "Phase8RealDevelopmentRunnerError",
    "Phase8RealDevelopmentSerializationError",
    "Phase8RealDevelopmentValidationError",
    "build_phase8_real_development_case_binding_collection",
    "build_phase8_real_development_case_bindings",
    "build_phase8_real_development_input_binding",
    "build_phase8_real_development_run_plan",
    "execute_phase8_real_development_run",
    "hash_phase8_real_development_case_binding",
    "hash_phase8_real_development_case_binding_collection",
    "hash_phase8_real_development_input_binding",
    "hash_phase8_real_development_run_plan",
    "phase8_real_development_case_binding_collection_from_mapping",
    "phase8_real_development_case_binding_collection_identity_payload",
    "phase8_real_development_case_binding_collection_to_dict",
    "phase8_real_development_case_binding_collection_to_json",
    "phase8_real_development_case_binding_from_mapping",
    "phase8_real_development_case_binding_identity_payload",
    "phase8_real_development_case_binding_to_dict",
    "phase8_real_development_input_binding_from_mapping",
    "phase8_real_development_input_binding_identity_payload",
    "phase8_real_development_input_binding_to_dict",
    "phase8_real_development_input_binding_to_json",
    "phase8_real_development_run_plan_from_mapping",
    "phase8_real_development_run_plan_identity_payload",
    "phase8_real_development_run_plan_to_dict",
    "phase8_real_development_run_plan_to_json",
    "run_phase8_real_development_plan_publication",
]
