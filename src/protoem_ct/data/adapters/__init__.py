"""Dataset adapter contracts for Phase 2 discovery."""

from protoem_ct.data.adapters.base import (
    AdapterCaseCandidate,
    AdapterInventory,
    AdapterLayoutSpec,
    DatasetAdapter,
    InvalidAdapterCandidateError,
    InvalidAdapterInventoryError,
    InvalidAdapterLayoutError,
    ResolvedAdapterCase,
    resolve_adapter_inventory_files,
)
from protoem_ct.data.adapters.lits import (
    LITS_STYLE_ADAPTER_NAME,
    LITS_STYLE_ADAPTER_VERSION,
    SUPPORTED_LITS_SUFFIXES,
    AmbiguousLiTSPairingError,
    IncompleteLiTSPairError,
    InvalidLiTSConventionError,
    LiTSAdapterError,
    LiTSDiscoveryError,
    LiTSFilenameConvention,
    LiTSStyleAdapter,
    MalformedLiTSFilenameError,
)

__all__ = [
    "AdapterCaseCandidate",
    "AdapterInventory",
    "AdapterLayoutSpec",
    "AmbiguousLiTSPairingError",
    "DatasetAdapter",
    "IncompleteLiTSPairError",
    "InvalidAdapterCandidateError",
    "InvalidAdapterInventoryError",
    "InvalidAdapterLayoutError",
    "InvalidLiTSConventionError",
    "LITS_STYLE_ADAPTER_NAME",
    "LITS_STYLE_ADAPTER_VERSION",
    "LiTSAdapterError",
    "LiTSDiscoveryError",
    "LiTSFilenameConvention",
    "LiTSStyleAdapter",
    "MalformedLiTSFilenameError",
    "ResolvedAdapterCase",
    "SUPPORTED_LITS_SUFFIXES",
    "resolve_adapter_inventory_files",
]
