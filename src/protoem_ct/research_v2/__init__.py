"""Research-v2 protocol contracts and fail-closed data guards."""

from protoem_ct.research_v2.protocol import (
    AccessPurpose,
    DataAccessViolation,
    DataPartition,
    PartitionIsolationError,
    PartitionIsolationReport,
    Phase2ArtifactIdentity,
    ProtocolValidationError,
    ResearchV2Protocol,
    VerifiedPhase2Artifacts,
    assert_data_access,
    assert_protocol_artifact_lock,
    audit_partition_isolation,
    build_nnunet_splits,
    canonical_json_sha256,
    load_protocol,
    load_verified_phase2_artifacts,
)

__all__ = [
    "AccessPurpose",
    "DataAccessViolation",
    "DataPartition",
    "PartitionIsolationError",
    "PartitionIsolationReport",
    "Phase2ArtifactIdentity",
    "ProtocolValidationError",
    "ResearchV2Protocol",
    "VerifiedPhase2Artifacts",
    "assert_data_access",
    "assert_protocol_artifact_lock",
    "audit_partition_isolation",
    "build_nnunet_splits",
    "canonical_json_sha256",
    "load_protocol",
    "load_verified_phase2_artifacts",
]
