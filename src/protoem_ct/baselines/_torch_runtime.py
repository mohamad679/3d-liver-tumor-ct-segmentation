"""Process-wide CPU PyTorch runtime configuration for Phase 3 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

PHASE3_TORCH_THREAD_FAILURE_CODES: frozenset[str] = frozenset({"torch_incompatible_thread_runtime"})
_REQUIRED_INTRA_OP_THREAD_COUNT = 1
_REQUIRED_INTER_OP_THREAD_COUNT = 1
_RUNTIME_CONFIGURATION_LOCK = Lock()
_INTEROP_CONFIGURATION_APPLIED = False


class Phase3TorchRuntimeError(ValueError):
    """Base error for invalid Phase 3 PyTorch runtime configuration."""


@dataclass(frozen=True, slots=True)
class Phase3TorchThreadConfigurationError(Phase3TorchRuntimeError):
    """Typed Phase 3 PyTorch thread-runtime failure."""

    failure_code: str
    intra_op_threads: int
    inter_op_threads: int

    def __post_init__(self) -> None:
        if self.failure_code not in PHASE3_TORCH_THREAD_FAILURE_CODES:
            raise Phase3TorchRuntimeError(
                f"Unsupported Phase 3 torch failure code: {self.failure_code!r}."
            )

    def __str__(self) -> str:
        return self.failure_code


def configure_phase3_cpu_torch_runtime(*, torch: Any) -> None:
    """Configure the process-wide CPU torch runtime for Phase 3 baselines."""

    global _INTEROP_CONFIGURATION_APPLIED

    required_intra = _REQUIRED_INTRA_OP_THREAD_COUNT
    required_inter = _REQUIRED_INTER_OP_THREAD_COUNT
    with _RUNTIME_CONFIGURATION_LOCK:
        current_intra = int(torch.get_num_threads())
        current_inter = int(torch.get_num_interop_threads())
        if _INTEROP_CONFIGURATION_APPLIED:
            _require_compatible_runtime(
                intra_op_threads=current_intra,
                inter_op_threads=current_inter,
            )
            return

        if current_intra == required_intra and current_inter == required_inter:
            _INTEROP_CONFIGURATION_APPLIED = True
            return

        torch.set_num_threads(required_intra)
        try:
            torch.set_num_interop_threads(required_inter)
        except RuntimeError as exc:
            updated_intra = int(torch.get_num_threads())
            updated_inter = int(torch.get_num_interop_threads())
            raise Phase3TorchThreadConfigurationError(
                failure_code="torch_incompatible_thread_runtime",
                intra_op_threads=updated_intra,
                inter_op_threads=updated_inter,
            ) from exc

        updated_intra = int(torch.get_num_threads())
        updated_inter = int(torch.get_num_interop_threads())
        _require_compatible_runtime(
            intra_op_threads=updated_intra,
            inter_op_threads=updated_inter,
        )
        _INTEROP_CONFIGURATION_APPLIED = True


def _require_compatible_runtime(*, intra_op_threads: int, inter_op_threads: int) -> None:
    if intra_op_threads != _REQUIRED_INTRA_OP_THREAD_COUNT:
        raise Phase3TorchThreadConfigurationError(
            failure_code="torch_incompatible_thread_runtime",
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
    if inter_op_threads != _REQUIRED_INTER_OP_THREAD_COUNT:
        raise Phase3TorchThreadConfigurationError(
            failure_code="torch_incompatible_thread_runtime",
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
