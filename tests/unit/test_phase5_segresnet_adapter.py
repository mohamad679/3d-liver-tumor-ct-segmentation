"""Unit tests for the Phase 5 SegResNet FeatureEncoder3D adapter."""

# mypy: disable-error-code=misc

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from protoem_ct.models import FeatureEncoder3D
from protoem_ct.models.segresnet_adapter import (
    SEGRESNET_FEATURE_ENCODER_IDENTITY,
    SEGRESNET_FEATURE_STAGE,
    SegResNetFeatureEncoder3D,
    SegResNetFeatureEncoderValidationError,
)

torch = pytest.importorskip("torch")
TorchModuleBase: Any = torch.nn.Module
TorchModuleList: Any = torch.nn.ModuleList


def _sha256(seed: str) -> str:
    return (seed * 64)[:64]


class AddConstant(TorchModuleBase):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.forward_call_count = 0
        self.grad_enabled_history: list[bool] = []

    def forward(self, x: Any) -> Any:
        self.forward_call_count += 1
        self.grad_enabled_history.append(torch.is_grad_enabled())
        return x + self.bias.view(1, 1, 1, 1, 1)


class MultiplyConstant(TorchModuleBase):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.forward_call_count = 0
        self.grad_enabled_history: list[bool] = []

    def forward(self, x: Any) -> Any:
        self.forward_call_count += 1
        self.grad_enabled_history.append(torch.is_grad_enabled())
        return x * self.weight.view(1, 1, 1, 1, 1)


class RaiseIfExecuted(TorchModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.forward_call_count = 0

    def forward(self, x: Any) -> Any:
        self.forward_call_count += 1
        raise AssertionError("Decoder or segmentation head should not execute.")


class ReturnNonFinite(TorchModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Any) -> Any:
        return torch.full_like(x, float("nan"))


class RaiseDuringEncoding(TorchModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Any) -> Any:
        raise RuntimeError("broken_down_layer")


class SqueezeRank(TorchModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Any) -> Any:
        return x.squeeze(0)


class FakeSegResNetLike(TorchModuleBase):
    def __init__(
        self,
        *,
        conv_init: Any | None = None,
        down_layers: Any | None = None,
        up_samples: Any | None = None,
        up_layers: Any | None = None,
        conv_final: Any | None = None,
    ) -> None:
        super().__init__()
        self.convInit = conv_init if conv_init is not None else AddConstant(1.0)
        self.down_layers = (
            down_layers
            if down_layers is not None
            else TorchModuleList((MultiplyConstant(2.0), AddConstant(3.0)))
        )
        self.up_samples = up_samples if up_samples is not None else RaiseIfExecuted()
        self.up_layers = up_layers if up_layers is not None else RaiseIfExecuted()
        self.conv_final = conv_final if conv_final is not None else RaiseIfExecuted()


@dataclass(frozen=True, slots=True)
class ExpectedFeature:
    tensor: Any
    batch_size: int
    channels: int
    spatial_shape: tuple[int, int, int]


def _volume() -> Any:
    return torch.arange(2 * 1 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4, 4)


def _expected_feature(model: FakeSegResNetLike, volume: Any) -> ExpectedFeature:
    features = model.down_layers[1](model.down_layers[0](model.convInit(volume)))
    return ExpectedFeature(
        tensor=features,
        batch_size=int(features.shape[0]),
        channels=int(features.shape[1]),
        spatial_shape=(int(features.shape[2]), int(features.shape[3]), int(features.shape[4])),
    )


def test_runtime_protocol_conformance() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())

    assert isinstance(adapter, FeatureEncoder3D)


def test_exact_final_encoder_boundary_extraction() -> None:
    model = FakeSegResNetLike()
    adapter = SegResNetFeatureEncoder3D(model=model)
    volume = _volume()

    result = adapter.encode(
        volume,
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    expected = _expected_feature(model, volume)
    feature_tensor = result.feature_data
    assert isinstance(feature_tensor, torch.Tensor)
    assert torch.equal(feature_tensor, expected.tensor)


def test_expected_output_shape_and_metadata_match_tensor_shape() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())
    volume = _volume()

    result = adapter.encode(
        volume,
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    feature_tensor = result.feature_data
    assert isinstance(feature_tensor, torch.Tensor)
    assert tuple(int(item) for item in feature_tensor.shape) == (2, 1, 4, 4, 4)
    assert result.metadata.encoder_identity == SEGRESNET_FEATURE_ENCODER_IDENTITY
    assert result.metadata.feature_stage == SEGRESNET_FEATURE_STAGE
    assert result.metadata.feature_channels == int(feature_tensor.shape[1])
    assert result.metadata.resolution.input_spatial_shape == (4, 4, 4)
    assert result.metadata.resolution.feature_spatial_shape == (4, 4, 4)
    assert result.metadata.resolution.downsample_factors == (1, 1, 1)


def test_repeated_encoding_is_deterministic() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())
    volume = _volume()

    first = adapter.encode(
        volume,
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )
    second = adapter.encode(
        volume,
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    assert isinstance(first.feature_data, torch.Tensor)
    assert isinstance(second.feature_data, torch.Tensor)
    assert torch.equal(first.feature_data, second.feature_data)
    assert first.metadata == second.metadata


def test_gradients_are_disabled_during_encoding() -> None:
    model = FakeSegResNetLike()
    adapter = SegResNetFeatureEncoder3D(model=model)

    adapter.encode(
        _volume(),
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    assert model.convInit.grad_enabled_history == [False]
    first_layer = model.down_layers[0]
    second_layer = model.down_layers[1]
    assert isinstance(first_layer, MultiplyConstant)
    assert isinstance(second_layer, AddConstant)
    assert first_layer.grad_enabled_history == [False]
    assert second_layer.grad_enabled_history == [False]


def test_caller_training_state_is_restored() -> None:
    model = FakeSegResNetLike()
    model.train(True)
    adapter = SegResNetFeatureEncoder3D(model=model)

    adapter.encode(
        _volume(),
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    assert model.training is True


def test_training_state_restored_after_exception() -> None:
    model = FakeSegResNetLike(down_layers=TorchModuleList((RaiseDuringEncoding(),)))
    model.train(True)
    adapter = SegResNetFeatureEncoder3D(model=model)

    with pytest.raises(RuntimeError, match="broken_down_layer"):
        adapter.encode(
            _volume(),
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )

    assert model.training is True


def test_invalid_rank_rejected() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())

    with pytest.raises(SegResNetFeatureEncoderValidationError, match="rank 5"):
        adapter.encode(
            torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )


def test_nonfloating_input_rejected() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())

    with pytest.raises(SegResNetFeatureEncoderValidationError, match="floating dtype"):
        adapter.encode(
            torch.zeros((1, 1, 4, 4, 4), dtype=torch.int64),
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )


def test_nonfinite_input_rejected() -> None:
    adapter = SegResNetFeatureEncoder3D(model=FakeSegResNetLike())
    volume = _volume()
    volume[0, 0, 0, 0, 0] = float("nan")

    with pytest.raises(SegResNetFeatureEncoderValidationError, match="finite values"):
        adapter.encode(
            volume,
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )


def test_unsupported_or_malformed_model_rejected() -> None:
    with pytest.raises(
        SegResNetFeatureEncoderValidationError,
        match="Unsupported baseline_family",
    ):
        SegResNetFeatureEncoder3D(
            model=FakeSegResNetLike(),
            baseline_family="nnunet_v2",
        )
    with pytest.raises(SegResNetFeatureEncoderValidationError, match="torch.nn.Module"):
        SegResNetFeatureEncoder3D(model=object())
    with pytest.raises(
        SegResNetFeatureEncoderValidationError,
        match="missing attribute 'down_layers'",
    ):
        malformed = FakeSegResNetLike()
        delattr(malformed, "down_layers")
        SegResNetFeatureEncoder3D(model=malformed)
    with pytest.raises(
        SegResNetFeatureEncoderValidationError,
        match="attribute 'convInit' must be a torch.nn.Module",
    ):
        malformed_type = FakeSegResNetLike()
        malformed_type.convInit = object()
        SegResNetFeatureEncoder3D(model=malformed_type)


def test_nonfinite_feature_rejected() -> None:
    adapter = SegResNetFeatureEncoder3D(
        model=FakeSegResNetLike(
            down_layers=TorchModuleList((ReturnNonFinite(),)),
        )
    )

    with pytest.raises(
        SegResNetFeatureEncoderValidationError,
        match="feature tensor must contain only finite",
    ):
        adapter.encode(
            _volume(),
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )


def test_malformed_feature_rank_rejected() -> None:
    adapter = SegResNetFeatureEncoder3D(
        model=FakeSegResNetLike(
            down_layers=TorchModuleList((SqueezeRank(),)),
        )
    )

    with pytest.raises(SegResNetFeatureEncoderValidationError, match="rank 5"):
        adapter.encode(
            _volume(),
            preprocessing_hash=_sha256("a"),
            checkpoint_hash=_sha256("b"),
            input_identity="query_case_001",
        )


def test_decoder_and_segmentation_head_not_executed() -> None:
    model = FakeSegResNetLike()
    adapter = SegResNetFeatureEncoder3D(model=model)

    adapter.encode(
        _volume(),
        preprocessing_hash=_sha256("a"),
        checkpoint_hash=_sha256("b"),
        input_identity="query_case_001",
    )

    assert model.up_samples.forward_call_count == 0
    assert model.up_layers.forward_call_count == 0
    assert model.conv_final.forward_call_count == 0
