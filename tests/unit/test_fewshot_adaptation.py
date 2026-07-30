"""Unit tests for deterministic Phase 4 adaptation-mode parameter selection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from protoem_ct.fewshot import (
    AdaptationParameterSelectionResult,
    FewshotAdaptationSelectionValidationError,
    apply_monai_segresnet_adaptation_mode,
)


@dataclass
class FakeParameter:
    """Minimal mutable parameter stand-in for adaptation tests."""

    element_count: int
    requires_grad: bool = True

    def numel(self) -> int:
        """Return the fixed scalar count for the fake parameter."""

        return self.element_count


class FakeModule:
    """Recursive fake module with deterministic ``named_parameters`` output."""

    def __init__(self) -> None:
        self._parameters: dict[str, FakeParameter] = {}
        self._modules: dict[str, FakeModule] = {}

    def add_parameter(self, name: str, element_count: int) -> None:
        """Add one parameter to this module."""

        self._parameters[name] = FakeParameter(element_count=element_count)

    def add_module(self, name: str, module: FakeModule) -> None:
        """Add one child module and expose it as an attribute."""

        self._modules[name] = module
        setattr(self, name, module)

    def named_parameters(self) -> list[tuple[str, FakeParameter]]:
        """Return deterministic recursive parameter names."""

        values: list[tuple[str, FakeParameter]] = []
        for name, parameter in self._parameters.items():
            values.append((name, parameter))
        for module_name, module in self._modules.items():
            for child_name, parameter in module.named_parameters():
                values.append((f"{module_name}.{child_name}", parameter))
        return values


class FakeSegResNet(FakeModule):
    """Tiny SegResNet-like test module with explicit encoder/decoder/head boundaries."""

    def __init__(self) -> None:
        super().__init__()
        conv_init = FakeModule()
        conv_init.add_parameter("weight", 10)
        down_layers = FakeModule()
        down_block0 = FakeModule()
        down_block0.add_parameter("weight", 20)
        down_block1 = FakeModule()
        down_block1.add_parameter("weight", 30)
        down_layers.add_module("0", down_block0)
        down_layers.add_module("1", down_block1)
        up_samples = FakeModule()
        up_samples.add_parameter("weight", 5)
        up_layers = FakeModule()
        up_block0 = FakeModule()
        up_block0.add_parameter("weight", 7)
        up_block1 = FakeModule()
        up_block1.add_parameter("weight", 11)
        up_layers.add_module("0", up_block0)
        up_layers.add_module("1", up_block1)
        conv_final = FakeModule()
        conv_final.add_parameter("weight", 13)
        conv_final.add_parameter("bias", 2)
        self.add_module("convInit", conv_init)
        self.add_module("down_layers", down_layers)
        self.add_module("up_samples", up_samples)
        self.add_module("up_layers", up_layers)
        self.add_module("conv_final", conv_final)


class MalformedSegResNetMissingHead(FakeSegResNet):
    """SegResNet-like test module missing the explicit head attribute."""

    def __init__(self) -> None:
        super().__init__()
        delattr(self, "conv_final")
        del self._modules["conv_final"]


class MalformedSegResNetAmbiguousHead(FakeSegResNet):
    """SegResNet-like test module with overlapping decoder/head module assignment."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_final = self._modules["up_layers"]


class MalformedSegResNetUncoveredParameter(FakeSegResNet):
    """SegResNet-like test module with an uncovered parameter outside the expected boundaries."""

    def __init__(self) -> None:
        super().__init__()
        auxiliary = FakeModule()
        auxiliary.add_parameter("weight", 3)
        self.add_module("auxiliary", auxiliary)


class SegResNetWithEmptyHead(FakeSegResNet):
    """SegResNet-like test module whose head boundary exposes zero parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.add_module("conv_final", FakeModule())


def _apply(
    model: FakeModule,
    adaptation_mode: str,
) -> AdaptationParameterSelectionResult:
    return apply_monai_segresnet_adaptation_mode(
        model,
        adaptation_mode=adaptation_mode,
        baseline_family="monai_segresnet",
    )


def test_head_only_freezes_encoder_and_decoder() -> None:
    model = FakeSegResNet()

    result = _apply(model, "head_only")

    assert result.trainable_parameter_names == ("conv_final.bias", "conv_final.weight")
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("conv_final")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("conv_final")
    )


def test_decoder_only_freezes_encoder_and_enables_decoder_plus_head() -> None:
    model = FakeSegResNet()

    result = _apply(model, "decoder_only")

    assert result.trainable_parameter_names == (
        "conv_final.bias",
        "conv_final.weight",
        "up_layers.0.weight",
        "up_layers.1.weight",
        "up_samples.weight",
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("convInit") or name.startswith("down_layers")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("up_samples")
        or name.startswith("up_layers")
        or name.startswith("conv_final")
    )


def test_full_finetune_enables_all_parameters() -> None:
    model = FakeSegResNet()

    result = _apply(model, "full_finetune")

    assert result.frozen_parameter_names == ()
    assert all(parameter.requires_grad for _name, parameter in model.named_parameters())


def test_exact_parameter_count_accounting() -> None:
    model = FakeSegResNet()

    head_only = _apply(model, "head_only")
    decoder_only = _apply(model, "decoder_only")
    full_finetune = _apply(model, "full_finetune")

    assert (
        head_only.total_parameter_count,
        head_only.trainable_parameter_count,
        head_only.frozen_parameter_count,
    ) == (98, 15, 83)
    assert (
        decoder_only.total_parameter_count,
        decoder_only.trainable_parameter_count,
        decoder_only.frozen_parameter_count,
    ) == (98, 38, 60)
    assert (
        full_finetune.total_parameter_count,
        full_finetune.trainable_parameter_count,
        full_finetune.frozen_parameter_count,
    ) == (98, 98, 0)


def test_deterministic_ordered_parameter_name_output() -> None:
    model = FakeSegResNet()

    result = _apply(model, "decoder_only")

    assert result.trainable_parameter_names == tuple(sorted(result.trainable_parameter_names))
    assert result.frozen_parameter_names == tuple(sorted(result.frozen_parameter_names))


def test_repeated_same_mode_application_is_identical() -> None:
    model = FakeSegResNet()

    first = _apply(model, "decoder_only")
    second = _apply(model, "decoder_only")

    assert first == second


def test_switching_modes_on_same_model_resets_requires_grad_state() -> None:
    model = FakeSegResNet()

    _apply(model, "full_finetune")
    second = _apply(model, "head_only")

    assert second.trainable_parameter_names == ("conv_final.bias", "conv_final.weight")
    assert second.frozen_parameter_count == 83
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("conv_final")
    )


def test_invalid_mode_rejected() -> None:
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        _apply(FakeSegResNet(), "encoder_only")


def test_unsupported_or_malformed_model_rejected() -> None:
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        apply_monai_segresnet_adaptation_mode(
            FakeSegResNet(),
            adaptation_mode="head_only",
            baseline_family="nnunet_v2",
        )
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        _apply(MalformedSegResNetMissingHead(), "head_only")
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        _apply(MalformedSegResNetAmbiguousHead(), "decoder_only")
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        _apply(MalformedSegResNetUncoveredParameter(), "full_finetune")


def test_zero_trainable_parameter_failure() -> None:
    with pytest.raises(FewshotAdaptationSelectionValidationError):
        _apply(SegResNetWithEmptyHead(), "head_only")
