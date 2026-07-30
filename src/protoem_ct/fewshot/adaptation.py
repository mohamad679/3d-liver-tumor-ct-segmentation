"""Deterministic Phase 4 adaptation-mode parameter selection and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from protoem_ct.fewshot.artifacts import SUPPORTED_ADAPTATION_MODES

SUPPORTED_ADAPTATION_BASELINE_FAMILIES: Final[frozenset[str]] = frozenset({"monai_segresnet"})
SEGRESNET_ENCODER_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = ("convInit", "down_layers")
SEGRESNET_DECODER_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = ("up_samples", "up_layers")
SEGRESNET_HEAD_ATTRIBUTE_NAME: Final[str] = "conv_final"


class FewshotAdaptationSelectionError(ValueError):
    """Base error for Phase 4 adaptation-mode parameter selection."""


class FewshotAdaptationSelectionValidationError(FewshotAdaptationSelectionError):
    """Raised when adaptation-mode selection inputs or results are invalid."""


@runtime_checkable
class ParameterLike(Protocol):
    """Minimal mutable parameter protocol required for adaptation selection."""

    requires_grad: bool

    def numel(self) -> int:
        """Return the number of scalar elements in the parameter."""


@runtime_checkable
class ModuleLike(Protocol):
    """Minimal module protocol required for explicit SegResNet boundary selection."""

    def named_parameters(self) -> object:
        """Yield parameter names relative to this module."""


@dataclass(frozen=True, slots=True)
class AdaptationParameterSelectionResult:
    """Immutable result of one deterministic adaptation-mode selection."""

    adaptation_mode: str
    total_parameter_count: int
    trainable_parameter_count: int
    frozen_parameter_count: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.adaptation_mode not in SUPPORTED_ADAPTATION_MODES:
            raise FewshotAdaptationSelectionValidationError(
                f"Unsupported adaptation_mode: {self.adaptation_mode!r}."
            )
        if self.total_parameter_count < 0:
            raise FewshotAdaptationSelectionValidationError(
                "total_parameter_count must be nonnegative."
            )
        if self.trainable_parameter_count <= 0:
            raise FewshotAdaptationSelectionValidationError(
                "trainable_parameter_count must be positive."
            )
        if self.frozen_parameter_count < 0:
            raise FewshotAdaptationSelectionValidationError(
                "frozen_parameter_count must be nonnegative."
            )
        if self.total_parameter_count != (
            self.trainable_parameter_count + self.frozen_parameter_count
        ):
            raise FewshotAdaptationSelectionValidationError(
                "total_parameter_count must equal "
                "trainable_parameter_count + frozen_parameter_count."
            )
        if tuple(sorted(self.trainable_parameter_names)) != self.trainable_parameter_names:
            raise FewshotAdaptationSelectionValidationError(
                "trainable_parameter_names must be in canonical sorted order."
            )
        if tuple(sorted(self.frozen_parameter_names)) != self.frozen_parameter_names:
            raise FewshotAdaptationSelectionValidationError(
                "frozen_parameter_names must be in canonical sorted order."
            )
        trainable_names = set(self.trainable_parameter_names)
        frozen_names = set(self.frozen_parameter_names)
        if trainable_names & frozen_names:
            raise FewshotAdaptationSelectionValidationError(
                "trainable_parameter_names and frozen_parameter_names must be disjoint."
            )
        if len(trainable_names) != len(self.trainable_parameter_names):
            raise FewshotAdaptationSelectionValidationError(
                "trainable_parameter_names must be unique."
            )
        if len(frozen_names) != len(self.frozen_parameter_names):
            raise FewshotAdaptationSelectionValidationError(
                "frozen_parameter_names must be unique."
            )


def apply_monai_segresnet_adaptation_mode(
    model: ModuleLike,
    *,
    adaptation_mode: str,
    baseline_family: str = "monai_segresnet",
) -> AdaptationParameterSelectionResult:
    """Apply one deterministic adaptation mode to a SegResNet-like model instance."""

    if baseline_family not in SUPPORTED_ADAPTATION_BASELINE_FAMILIES:
        raise FewshotAdaptationSelectionValidationError(
            f"Unsupported baseline_family: {baseline_family!r}."
        )
    if adaptation_mode not in SUPPORTED_ADAPTATION_MODES:
        raise FewshotAdaptationSelectionValidationError(
            f"Unsupported adaptation_mode: {adaptation_mode!r}."
        )
    root_parameters = _collect_named_parameters(model, owner_name="model")
    if not root_parameters:
        raise FewshotAdaptationSelectionValidationError("Model must expose at least one parameter.")

    encoder_parameters = _collect_group_parameters(
        model,
        attribute_names=SEGRESNET_ENCODER_ATTRIBUTE_NAMES,
        group_name="encoder",
    )
    decoder_parameters = _collect_group_parameters(
        model,
        attribute_names=SEGRESNET_DECODER_ATTRIBUTE_NAMES,
        group_name="decoder",
    )
    head_parameters = _collect_group_parameters(
        model,
        attribute_names=(SEGRESNET_HEAD_ATTRIBUTE_NAME,),
        group_name="head",
    )

    _validate_group_partition(
        root_parameters=root_parameters,
        encoder_parameters=encoder_parameters,
        decoder_parameters=decoder_parameters,
        head_parameters=head_parameters,
    )

    selected_names: set[str]
    if adaptation_mode == "head_only":
        selected_names = set(head_parameters)
    elif adaptation_mode == "decoder_only":
        selected_names = set(decoder_parameters) | set(head_parameters)
    else:
        selected_names = set(root_parameters)

    for parameter in root_parameters.values():
        parameter.requires_grad = False
    for name in selected_names:
        root_parameters[name].requires_grad = True

    trainable_names = tuple(
        sorted(name for name, parameter in root_parameters.items() if parameter.requires_grad)
    )
    frozen_names = tuple(
        sorted(name for name, parameter in root_parameters.items() if not parameter.requires_grad)
    )
    if not trainable_names:
        raise FewshotAdaptationSelectionValidationError(
            "Selected adaptation mode left zero trainable parameters."
        )
    total_parameter_count = sum(
        _parameter_numel(parameter, field_name=name) for name, parameter in root_parameters.items()
    )
    trainable_parameter_count = sum(
        _parameter_numel(root_parameters[name], field_name=name) for name in trainable_names
    )
    frozen_parameter_count = sum(
        _parameter_numel(root_parameters[name], field_name=name) for name in frozen_names
    )
    return AdaptationParameterSelectionResult(
        adaptation_mode=adaptation_mode,
        total_parameter_count=total_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        frozen_parameter_count=frozen_parameter_count,
        trainable_parameter_names=trainable_names,
        frozen_parameter_names=frozen_names,
    )


def _collect_group_parameters(
    model: ModuleLike,
    *,
    attribute_names: tuple[str, ...],
    group_name: str,
) -> dict[str, ParameterLike]:
    parameters: dict[str, ParameterLike] = {}
    for attribute_name in attribute_names:
        if not hasattr(model, attribute_name):
            raise FewshotAdaptationSelectionValidationError(
                f"SegResNet {group_name} boundary is missing attribute {attribute_name!r}."
            )
        module = getattr(model, attribute_name)
        if not isinstance(module, ModuleLike):
            raise FewshotAdaptationSelectionValidationError(
                f"SegResNet attribute {attribute_name!r} is not module-like."
            )
        scoped_parameters = _collect_named_parameters(module, owner_name=attribute_name)
        if not scoped_parameters:
            continue
        for name, parameter in scoped_parameters.items():
            qualified_name = f"{attribute_name}.{name}"
            if qualified_name in parameters:
                raise FewshotAdaptationSelectionValidationError(
                    "Duplicate parameter assignment detected for "
                    f"{qualified_name!r} in {group_name}."
                )
            parameters[qualified_name] = parameter
    return parameters


def _collect_named_parameters(
    module: ModuleLike,
    *,
    owner_name: str,
) -> dict[str, ParameterLike]:
    raw_named_parameters = getattr(module, "named_parameters", None)
    if raw_named_parameters is None or not callable(raw_named_parameters):
        raise FewshotAdaptationSelectionValidationError(
            f"{owner_name} must define a callable named_parameters method."
        )
    try:
        values = list(raw_named_parameters())
    except Exception as error:
        raise FewshotAdaptationSelectionValidationError(
            f"{owner_name}.named_parameters() failed: {error}"
        ) from error
    parameters: dict[str, ParameterLike] = {}
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise FewshotAdaptationSelectionValidationError(
                f"{owner_name}.named_parameters() must yield (name, parameter) pairs."
            )
        name, parameter = item
        if not isinstance(name, str) or not name:
            raise FewshotAdaptationSelectionValidationError(
                f"{owner_name}.named_parameters() produced an invalid parameter name."
            )
        if name in parameters:
            raise FewshotAdaptationSelectionValidationError(
                f"{owner_name}.named_parameters() produced duplicate parameter name {name!r}."
            )
        _parameter_numel(parameter, field_name=f"{owner_name}.{name}")
        parameters[name] = parameter
    return parameters


def _parameter_numel(parameter: object, *, field_name: str) -> int:
    if not hasattr(parameter, "numel") or not callable(parameter.numel):
        raise FewshotAdaptationSelectionValidationError(
            f"{field_name} must expose a callable numel() method."
        )
    value = parameter.numel()
    if not isinstance(value, int) or value < 0:
        raise FewshotAdaptationSelectionValidationError(
            f"{field_name}.numel() must return a nonnegative integer."
        )
    if not hasattr(parameter, "requires_grad"):
        raise FewshotAdaptationSelectionValidationError(
            f"{field_name} must expose a requires_grad attribute."
        )
    return value


def _validate_group_partition(
    *,
    root_parameters: dict[str, ParameterLike],
    encoder_parameters: dict[str, ParameterLike],
    decoder_parameters: dict[str, ParameterLike],
    head_parameters: dict[str, ParameterLike],
) -> None:
    encoder_names = set(encoder_parameters)
    decoder_names = set(decoder_parameters)
    head_names = set(head_parameters)
    if encoder_names & decoder_names:
        raise FewshotAdaptationSelectionValidationError(
            "Encoder and decoder parameter assignments must be disjoint."
        )
    if encoder_names & head_names:
        raise FewshotAdaptationSelectionValidationError(
            "Encoder and head parameter assignments must be disjoint."
        )
    if decoder_names & head_names:
        raise FewshotAdaptationSelectionValidationError(
            "Decoder and head parameter assignments must be disjoint."
        )
    covered_names = encoder_names | decoder_names | head_names
    root_names = set(root_parameters)
    if covered_names != root_names:
        missing_names = tuple(sorted(root_names - covered_names))
        extra_names = tuple(sorted(covered_names - root_names))
        details: list[str] = []
        if missing_names:
            details.append(f"uncovered={missing_names!r}")
        if extra_names:
            details.append(f"unknown={extra_names!r}")
        raise FewshotAdaptationSelectionValidationError(
            "SegResNet boundary mapping must cover every root parameter exactly once"
            + (f": {', '.join(details)}." if details else ".")
        )


__all__ = [
    "AdaptationParameterSelectionResult",
    "FewshotAdaptationSelectionError",
    "FewshotAdaptationSelectionValidationError",
    "SEGRESNET_DECODER_ATTRIBUTE_NAMES",
    "SEGRESNET_ENCODER_ATTRIBUTE_NAMES",
    "SEGRESNET_HEAD_ATTRIBUTE_NAME",
    "SUPPORTED_ADAPTATION_BASELINE_FAMILIES",
    "apply_monai_segresnet_adaptation_mode",
]
