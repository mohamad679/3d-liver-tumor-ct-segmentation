from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import protoem_ct.baselines._torch_runtime as torch_runtime
from protoem_ct.baselines._torch_runtime import (
    Phase3TorchThreadConfigurationError,
    configure_phase3_cpu_torch_runtime,
)
from protoem_ct.baselines.monai_segresnet import _configure_torch_runtime as configure_monai_runtime
from protoem_ct.baselines.nnunet_tiny import _configure_torch_runtime as configure_nnunet_runtime


@dataclass
class _FakeTorch:
    intra_op_threads: int
    inter_op_threads: int
    raise_on_set_interop: bool = False
    set_num_threads_calls: int = 0
    set_num_interop_threads_calls: int = 0

    def set_num_threads(self, value: int) -> None:
        self.set_num_threads_calls += 1
        self.intra_op_threads = value

    def get_num_threads(self) -> int:
        return self.intra_op_threads

    def set_num_interop_threads(self, value: int) -> None:
        self.set_num_interop_threads_calls += 1
        if self.raise_on_set_interop:
            raise RuntimeError("parallel work already started")
        self.inter_op_threads = value

    def get_num_interop_threads(self) -> int:
        return self.inter_op_threads


@pytest.fixture(autouse=True)
def _reset_torch_runtime_state() -> None:
    torch_runtime._INTEROP_CONFIGURATION_APPLIED = False


def test_first_configuration_sets_intra_and_interop_to_one() -> None:
    fake_torch = _FakeTorch(intra_op_threads=8, inter_op_threads=4)

    configure_phase3_cpu_torch_runtime(torch=fake_torch)

    assert fake_torch.get_num_threads() == 1
    assert fake_torch.get_num_interop_threads() == 1
    assert fake_torch.set_num_threads_calls == 1
    assert fake_torch.set_num_interop_threads_calls == 1


def test_repeated_identical_configuration_is_idempotent() -> None:
    fake_torch = _FakeTorch(intra_op_threads=8, inter_op_threads=4)

    configure_phase3_cpu_torch_runtime(torch=fake_torch)
    configure_phase3_cpu_torch_runtime(torch=fake_torch)

    assert fake_torch.set_num_threads_calls == 1
    assert fake_torch.set_num_interop_threads_calls == 1
    assert fake_torch.get_num_threads() == 1
    assert fake_torch.get_num_interop_threads() == 1


def test_sequential_baseline_order_succeeds() -> None:
    fake_torch = _FakeTorch(intra_op_threads=8, inter_op_threads=4)

    configure_nnunet_runtime(torch=fake_torch)
    configure_monai_runtime(torch=fake_torch)

    assert fake_torch.set_num_interop_threads_calls == 1
    assert fake_torch.get_num_threads() == 1
    assert fake_torch.get_num_interop_threads() == 1


def test_reverse_baseline_order_succeeds() -> None:
    fake_torch = _FakeTorch(intra_op_threads=8, inter_op_threads=4)

    configure_monai_runtime(torch=fake_torch)
    configure_nnunet_runtime(torch=fake_torch)

    assert fake_torch.set_num_interop_threads_calls == 1
    assert fake_torch.get_num_threads() == 1
    assert fake_torch.get_num_interop_threads() == 1


def test_existing_compatible_runtime_is_accepted() -> None:
    fake_torch = _FakeTorch(intra_op_threads=1, inter_op_threads=1, raise_on_set_interop=True)

    configure_phase3_cpu_torch_runtime(torch=fake_torch)

    assert fake_torch.set_num_threads_calls == 0
    assert fake_torch.set_num_interop_threads_calls == 0


def test_existing_incompatible_runtime_raises_typed_failure() -> None:
    fake_torch = _FakeTorch(intra_op_threads=8, inter_op_threads=4, raise_on_set_interop=True)

    with pytest.raises(Phase3TorchThreadConfigurationError) as exc_info:
        configure_phase3_cpu_torch_runtime(torch=fake_torch)

    assert exc_info.value.failure_code == "torch_incompatible_thread_runtime"
    assert fake_torch.set_num_interop_threads_calls == 1


def test_fresh_subprocess_runtime_initialization_succeeds() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    baseline_python = (
        repository_root / "environments/phase3-baselines/intel-macos-cpu/.venv/bin/python"
    )
    script = """
from protoem_ct.baselines._torch_runtime import configure_phase3_cpu_torch_runtime
import torch

configure_phase3_cpu_torch_runtime(torch=torch)
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
value = torch.arange(8, dtype=torch.float32).reshape(2, 4).sum().item()
assert value == 28.0
configure_phase3_cpu_torch_runtime(torch=torch)
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repository_root / "src")

    result = subprocess.run(
        [str(baseline_python), "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=repository_root,
        env=env,
    )

    assert result.returncode == 0, result.stderr
