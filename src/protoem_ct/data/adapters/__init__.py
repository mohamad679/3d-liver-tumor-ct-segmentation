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

__all__ = [
    "AdapterCaseCandidate",
    "AdapterInventory",
    "AdapterLayoutSpec",
    "DatasetAdapter",
    "InvalidAdapterCandidateError",
    "InvalidAdapterInventoryError",
    "InvalidAdapterLayoutError",
    "ResolvedAdapterCase",
    "resolve_adapter_inventory_files",
]
