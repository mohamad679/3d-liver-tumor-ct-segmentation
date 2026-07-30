"""SegResNet FeatureEncoder3D adapter for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

from protoem_ct.models.interfaces import (
    EmbeddingMetadata,
    FeatureEncoder3D,
    FeatureEncoding3D,
    FeatureResolution3D,
    RetrievalContractValidationError,
)

SEGRESNET_FEATURE_ENCODER_IDENTITY: Final[str] = "monai_segresnet_final_encoder_v1"
SEGRESNET_FEATURE_STAGE: Final[str] = "final_encoder"
SUPPORTED_SEGRESNET_ADAPTER_BASELINE_FAMILIES: Final[frozenset[str]] = frozenset(
    {"monai_segresnet"}
)


class SegResNetFeatureEncoderError(ValueError):
    """Base error for the Phase 5 SegResNet feature adapter."""


class SegResNetFeatureEncoderValidationError(SegResNetFeatureEncoderError):
    """Raised when the SegResNet feature adapter input or output is invalid."""


@dataclass(frozen=True, slots=True)
class SegResNetFeatureEncoder3D(FeatureEncoder3D):
    """Concrete deterministic Phase 5 feature encoder for SegResNet-like models."""

    model: object
    baseline_family: str = "monai_segresnet"
    encoder_identity: str = SEGRESNET_FEATURE_ENCODER_IDENTITY

    def __post_init__(self) -> None:
        _require_torch_available()
        torch = cast(Any, _import_torch())
        if self.baseline_family not in SUPPORTED_SEGRESNET_ADAPTER_BASELINE_FAMILIES:
            raise SegResNetFeatureEncoderValidationError(
                f"Unsupported baseline_family: {self.baseline_family!r}."
            )
        if not isinstance(self.model, torch.nn.Module):
            raise SegResNetFeatureEncoderValidationError(
                "model must be a torch.nn.Module for the SegResNet feature adapter."
            )
        _require_encoder_boundaries(cast(object, self.model), torch=torch)

    def encode(
        self,
        volume: object,
        *,
        preprocessing_hash: str,
        checkpoint_hash: str,
        input_identity: str,
    ) -> FeatureEncoding3D:
        """Encode one 5D model-compatible tensor to the final encoder feature map."""

        torch = cast(Any, _import_torch())
        model = cast(Any, self.model)
        tensor = cast(Any, _validate_input_tensor(volume, torch=torch))
        _validate_device_match(model=model, tensor=tensor, torch=torch)

        original_training = bool(model.training)
        try:
            model.eval()
            with torch.no_grad():
                conv_init = cast(
                    Any,
                    _require_module_boundary(
                        model,
                        attribute_name="convInit",
                        torch=torch,
                    ),
                )
                down_layers = cast(
                    Any,
                    _require_module_boundary(
                        model,
                        attribute_name="down_layers",
                        torch=torch,
                    ),
                )
                features = conv_init(tensor)
                features = _apply_down_layers(
                    features=features,
                    down_layers=down_layers,
                    torch=torch,
                )
        except RetrievalContractValidationError as error:
            raise SegResNetFeatureEncoderValidationError(str(error)) from error
        finally:
            model.train(original_training)

        if not isinstance(features, torch.Tensor):
            raise SegResNetFeatureEncoderValidationError(
                "Final encoder boundary must return a torch.Tensor feature map."
            )
        if features.ndim != 5:
            raise SegResNetFeatureEncoderValidationError(
                "Final encoder feature tensor must have rank 5 [B, C, D, H, W]."
            )
        if not torch.is_floating_point(features):
            raise SegResNetFeatureEncoderValidationError(
                "Final encoder feature tensor must use a floating dtype."
            )
        if not torch.isfinite(features).all().item():
            raise SegResNetFeatureEncoderValidationError(
                "Final encoder feature tensor must contain only finite values."
            )
        if features.shape[0] != tensor.shape[0]:
            raise SegResNetFeatureEncoderValidationError(
                "Final encoder feature tensor must preserve the batch dimension."
            )

        input_spatial_shape = _spatial_shape_from_tensor(tensor)
        feature_spatial_shape = _spatial_shape_from_tensor(features)

        metadata = EmbeddingMetadata(
            encoder_identity=self.encoder_identity,
            preprocessing_hash=preprocessing_hash,
            checkpoint_hash=checkpoint_hash,
            input_identity=input_identity,
            feature_stage=SEGRESNET_FEATURE_STAGE,
            normalization_name="unscaled_raw",
            feature_channels=int(features.shape[1]),
            resolution=FeatureResolution3D(
                input_spatial_shape=input_spatial_shape,
                feature_spatial_shape=feature_spatial_shape,
                downsample_factors=_derive_downsample_factors(
                    input_shape=input_spatial_shape,
                    feature_shape=feature_spatial_shape,
                ),
            ),
        )
        if metadata.feature_channels != int(features.shape[1]):
            raise SegResNetFeatureEncoderValidationError(
                "Embedding metadata feature_channels does not match the encoded tensor."
            )
        if metadata.resolution.feature_spatial_shape != tuple(
            int(item) for item in features.shape[2:]
        ):
            raise SegResNetFeatureEncoderValidationError(
                "Embedding metadata feature_spatial_shape does not match the encoded tensor."
            )
        return FeatureEncoding3D(feature_data=features, metadata=metadata)


def _apply_down_layers(*, features: object, down_layers: object, torch: Any) -> object:
    module = cast(Any, down_layers)
    if isinstance(module, torch.nn.ModuleList):
        result = features
        for index, layer in enumerate(module):
            if not isinstance(layer, torch.nn.Module):
                raise SegResNetFeatureEncoderValidationError(
                    f"down_layers[{index}] is not a torch.nn.Module."
                )
            result = layer(result)
        return result
    if isinstance(module, torch.nn.Sequential):
        return module(features)
    try:
        children = list(module.children())
    except Exception:
        children = []
    if children:
        result = features
        for index, layer in enumerate(children):
            if not isinstance(layer, torch.nn.Module):
                raise SegResNetFeatureEncoderValidationError(
                    f"down_layers child {index} is not a torch.nn.Module."
                )
            result = layer(result)
        return result
    if not callable(module):
        raise SegResNetFeatureEncoderValidationError(
            "down_layers must be callable or iterable over callable module children."
        )
    return module(features)


def _require_encoder_boundaries(model: object, *, torch: Any) -> None:
    _require_module_boundary(model, attribute_name="convInit", torch=torch)
    _require_module_boundary(model, attribute_name="down_layers", torch=torch)


def _require_module_boundary(model: object, *, attribute_name: str, torch: Any) -> object:
    if not hasattr(model, attribute_name):
        raise SegResNetFeatureEncoderValidationError(
            f"SegResNet encoder boundary is missing attribute {attribute_name!r}."
        )
    value = getattr(model, attribute_name)
    if not isinstance(value, torch.nn.Module):
        raise SegResNetFeatureEncoderValidationError(
            f"SegResNet attribute {attribute_name!r} must be a torch.nn.Module."
        )
    return value


def _validate_input_tensor(volume: object, *, torch: Any) -> Any:
    if not isinstance(volume, torch.Tensor):
        raise SegResNetFeatureEncoderValidationError("volume must be a torch.Tensor.")
    if volume.ndim != 5:
        raise SegResNetFeatureEncoderValidationError("volume must have rank 5 [B, C, D, H, W].")
    if not torch.is_floating_point(volume):
        raise SegResNetFeatureEncoderValidationError("volume must use a floating dtype.")
    if not torch.isfinite(volume).all().item():
        raise SegResNetFeatureEncoderValidationError("volume must contain only finite values.")
    return volume


def _validate_device_match(*, model: object, tensor: object, torch: Any) -> None:
    module = cast(Any, model)
    parameter_devices = {parameter.device for parameter in module.parameters()}
    if not parameter_devices:
        return
    if len(parameter_devices) != 1:
        raise SegResNetFeatureEncoderValidationError(
            "SegResNet feature adapter requires all model parameters on one device."
        )
    (model_device,) = tuple(parameter_devices)
    tensor_value = cast(Any, tensor)
    if tensor_value.device != model_device:
        raise SegResNetFeatureEncoderValidationError(
            "volume device must match the model parameter device exactly."
        )


def _spatial_shape_from_tensor(tensor: object) -> tuple[int, int, int]:
    value = cast(Any, tensor)
    return (
        int(value.shape[2]),
        int(value.shape[3]),
        int(value.shape[4]),
    )


def _derive_downsample_factors(
    *,
    input_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    values: list[int] = []
    for axis, input_size, feature_size in zip(
        ("depth", "height", "width"),
        input_shape,
        feature_shape,
        strict=True,
    ):
        if feature_size <= 0:
            raise SegResNetFeatureEncoderValidationError(f"feature_shape.{axis} must be positive.")
        if input_size % feature_size != 0:
            raise SegResNetFeatureEncoderValidationError(
                f"input_shape.{axis} must be divisible by feature_shape.{axis}."
            )
        values.append(input_size // feature_size)
    return (values[0], values[1], values[2])


def _require_torch_available() -> None:
    _import_torch()


def _import_torch() -> object:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise SegResNetFeatureEncoderValidationError(
            "torch is required for the SegResNet feature adapter."
        ) from error
    return torch


__all__ = [
    "SEGRESNET_FEATURE_ENCODER_IDENTITY",
    "SEGRESNET_FEATURE_STAGE",
    "SUPPORTED_SEGRESNET_ADAPTER_BASELINE_FAMILIES",
    "SegResNetFeatureEncoder3D",
    "SegResNetFeatureEncoderError",
    "SegResNetFeatureEncoderValidationError",
]
