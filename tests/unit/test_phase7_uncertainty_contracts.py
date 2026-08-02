from __future__ import annotations

import hashlib
from dataclasses import fields

import numpy as np
import pytest

from protoem_ct.uncertainty.contracts import (
    BinaryPredictiveProbabilityMap,
    Phase7UncertaintyContractError,
    PredictiveEntropyInput,
    PredictiveEntropyResultMetadata,
    ProbabilityMapValidationError,
    TTAContractValidationError,
    TTAProbabilitySamples,
    TTASampleManifest,
    TTASampleRecord,
    UncertaintyEvaluationEligibilityRecord,
    UncertaintyIdentityMismatchError,
    UncertaintySummaryInput,
    build_binary_predictive_probability_map,
    build_predictive_entropy_input,
    build_predictive_entropy_result_metadata,
    build_tta_probability_samples,
    build_tta_sample_manifest,
    build_tta_sample_record,
    build_uncertainty_evaluation_eligibility_record,
    build_uncertainty_summary_input,
    query_label_not_part_of_uncertainty_contracts_api,
)


def test_valid_probability_contract_is_deterministic() -> None:
    fg = _foreground_probabilities()
    bg = 1.0 - fg

    probability_map = build_binary_predictive_probability_map(
        foreground_probability_map=fg,
        background_probability_map=bg,
        source_prediction_identity_hash=_sha("phase6-inference"),
        common_grid_identity_hash=_sha("common-grid"),
    )
    repeated = build_binary_predictive_probability_map(
        foreground_probability_map=np.asfortranarray(fg),
        background_probability_map=np.asfortranarray(bg),
        source_prediction_identity_hash=_sha("phase6-inference"),
        common_grid_identity_hash=_sha("common-grid"),
    )

    assert probability_map.probability_shape == (1, 1, 2, 2, 2)
    assert probability_map.probability_map_identity_hash == repeated.probability_map_identity_hash
    assert np.array_equal(probability_map.foreground_probability_map, fg)
    assert np.array_equal(probability_map.background_probability_map, bg)


def test_invalid_probability_shape_is_rejected() -> None:
    fg = np.array([0.2, 0.8], dtype=np.float64)
    bg = 1.0 - fg

    with pytest.raises(ProbabilityMapValidationError, match=r"\[1,1,D,H,W\]"):
        build_binary_predictive_probability_map(
            foreground_probability_map=fg,
            background_probability_map=bg,
        )


def test_nonfinite_probabilities_are_rejected() -> None:
    fg = _foreground_probabilities()
    fg[0, 0, 0, 0, 0] = np.nan

    with pytest.raises(ProbabilityMapValidationError, match="finite"):
        build_binary_predictive_probability_map(
            foreground_probability_map=fg,
            background_probability_map=np.ones_like(fg) - fg,
        )


def test_out_of_range_probabilities_are_rejected() -> None:
    fg = _foreground_probabilities()
    fg[0, 0, 0, 0, 0] = 1.2

    with pytest.raises(ProbabilityMapValidationError, match=r"\[0,1\]"):
        build_binary_predictive_probability_map(
            foreground_probability_map=fg,
            background_probability_map=np.zeros_like(fg),
        )


def test_probability_pair_sum_mismatch_is_rejected() -> None:
    fg = _foreground_probabilities()
    bg = np.full_like(fg, 0.5)

    with pytest.raises(ProbabilityMapValidationError, match="sum to one"):
        build_binary_predictive_probability_map(
            foreground_probability_map=fg,
            background_probability_map=bg,
        )


def test_entropy_input_and_result_metadata_contracts_do_not_compute_entropy() -> None:
    probability_map = _probability_map("p0")
    entropy_input = build_predictive_entropy_input(probability_map)
    metadata = build_predictive_entropy_result_metadata(
        entropy_input=entropy_input,
        entropy_map_content_hash=_sha("entropy-map-content"),
        valid_voxel_count=8,
    )

    assert entropy_input.log_base == "natural"
    assert entropy_input.zero_probability_policy == "zero_log_zero_is_zero"
    assert metadata.entropy_shape == probability_map.probability_shape
    assert metadata.valid_voxel_count == 8


def test_tta_sample_count_validation_requires_two_samples() -> None:
    probability_map = _probability_map("p0")
    record = _sample_record(0, probability_map)
    manifest = build_tta_sample_manifest((record,))

    with pytest.raises(TTAContractValidationError, match="at least two samples"):
        build_tta_probability_samples(
            sample_manifest=manifest,
            probability_maps=(probability_map,),
        )


def test_tta_manifest_order_and_probability_map_order_are_validated() -> None:
    first = _probability_map("p0", offset=0.0)
    second = _probability_map("p1", offset=0.05)
    first_record = _sample_record(0, first)
    second_record = _sample_record(1, second)
    manifest = build_tta_sample_manifest((first_record, second_record))

    samples = build_tta_probability_samples(
        sample_manifest=manifest,
        probability_maps=(first, second),
    )

    assert samples.sample_count == 2
    assert samples.variance_population_policy == "population_variance"

    with pytest.raises(TTAContractValidationError, match="manifest order"):
        build_tta_probability_samples(
            sample_manifest=manifest,
            probability_maps=(second, first),
        )


def test_tta_manifest_rejects_noncontiguous_indices() -> None:
    first = _probability_map("p0", offset=0.0)
    second = _probability_map("p1", offset=0.05)
    first_record = _sample_record(0, first)
    bad_second_record = build_tta_sample_record(
        sample_index=2,
        sample_id="sample-2",
        probability_map_identity_hash=second.probability_map_identity_hash,
        common_grid_identity_hash=_sha("common-grid"),
        transform_manifest_hash=_sha("transform-2"),
    )

    with pytest.raises(TTAContractValidationError, match="zero-based, contiguous"):
        build_tta_sample_manifest((first_record, bad_second_record))


def test_uncertainty_summary_input_validates_required_source_links() -> None:
    probability_map = _probability_map("p0")
    summary_input = build_uncertainty_summary_input(
        uncertainty_source_type="predictive_entropy",
        uncertainty_map_content_hash=_sha("entropy-content"),
        uncertainty_shape=probability_map.probability_shape,
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        valid_voxel_count=8,
    )

    assert summary_input.uncertainty_source_type == "predictive_entropy"

    with pytest.raises(Phase7UncertaintyContractError, match="probability_map_identity_hash"):
        build_uncertainty_summary_input(
            uncertainty_source_type="predictive_entropy",
            uncertainty_map_content_hash=_sha("entropy-content"),
            uncertainty_shape=probability_map.probability_shape,
            valid_voxel_count=8,
        )


def test_evaluation_eligibility_records_validate_status_consistency() -> None:
    eligible = build_uncertainty_evaluation_eligibility_record(
        evaluation_name="tta_variance",
        status="eligible",
        valid_voxel_count=8,
        sample_count=2,
    )
    unavailable = build_uncertainty_evaluation_eligibility_record(
        evaluation_name="tta_variance",
        status="unavailable",
        valid_voxel_count=8,
        sample_count=1,
        reason_code="insufficient_samples",
        reason_message="TTA variance requires at least two samples.",
    )

    assert eligible.status == "eligible"
    assert unavailable.status == "unavailable"

    with pytest.raises(Phase7UncertaintyContractError, match="reason_code"):
        build_uncertainty_evaluation_eligibility_record(
            evaluation_name="predictive_entropy",
            status="unavailable",
            valid_voxel_count=8,
        )

    with pytest.raises(Phase7UncertaintyContractError, match="sample_count >= 2"):
        build_uncertainty_evaluation_eligibility_record(
            evaluation_name="tta_variance",
            status="eligible",
            valid_voxel_count=8,
            sample_count=1,
        )


def test_probability_self_identity_mismatch_is_rejected() -> None:
    probability_map = _probability_map("p0")

    with pytest.raises(UncertaintyIdentityMismatchError):
        BinaryPredictiveProbabilityMap(
            schema_name=probability_map.schema_name,
            schema_version=probability_map.schema_version,
            probability_map_identity_hash=_sha("wrong"),
            foreground_probability_map=probability_map.foreground_probability_map,
            background_probability_map=probability_map.background_probability_map,
            foreground_probability_content_hash=(
                probability_map.foreground_probability_content_hash
            ),
            background_probability_content_hash=(
                probability_map.background_probability_content_hash
            ),
            probability_shape=probability_map.probability_shape,
            probability_sum_tolerance=probability_map.probability_sum_tolerance,
            source_prediction_identity_hash=probability_map.source_prediction_identity_hash,
            common_grid_identity_hash=probability_map.common_grid_identity_hash,
        )


def test_query_labels_and_reference_masks_are_absent_from_public_contracts() -> None:
    public_contracts = (
        BinaryPredictiveProbabilityMap,
        PredictiveEntropyInput,
        PredictiveEntropyResultMetadata,
        TTASampleRecord,
        TTASampleManifest,
        TTAProbabilitySamples,
        UncertaintySummaryInput,
        UncertaintyEvaluationEligibilityRecord,
    )

    for contract in public_contracts:
        field_names = {field.name for field in fields(contract)}
        assert "query_label" not in field_names
        assert "query_labels" not in field_names
        assert "label_map" not in field_names
        assert "reference_mask" not in field_names
        assert "query_reference_mask" not in field_names

    assert query_label_not_part_of_uncertainty_contracts_api()


def _foreground_probabilities() -> np.ndarray:
    return np.array(
        [[[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]]],
        dtype=np.float64,
    )


def _probability_map(identifier: str, *, offset: float = 0.0) -> BinaryPredictiveProbabilityMap:
    fg = np.clip(_foreground_probabilities() + offset, 0.05, 0.95)
    return build_binary_predictive_probability_map(
        foreground_probability_map=fg,
        background_probability_map=1.0 - fg,
        source_prediction_identity_hash=_sha(f"prediction-{identifier}"),
        common_grid_identity_hash=_sha("common-grid"),
    )


def _sample_record(
    sample_index: int,
    probability_map: BinaryPredictiveProbabilityMap,
) -> TTASampleRecord:
    return build_tta_sample_record(
        sample_index=sample_index,
        sample_id=f"sample-{sample_index}",
        probability_map_identity_hash=probability_map.probability_map_identity_hash,
        common_grid_identity_hash=_sha("common-grid"),
        transform_manifest_hash=_sha(f"transform-{sample_index}"),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
