"""Leakage and reference-mask isolation tests for Phase 5 comparison execution."""

from __future__ import annotations

import numpy as np
import pytest

from protoem_ct.retrieval import (
    Phase5ComparisonLeakageError,
    Phase5ComparisonValidationError,
    build_phase5_synthetic_fixture,
    run_phase5_comparison,
)


def test_reference_mask_isolation_from_method_execution() -> None:
    fixture = build_phase5_synthetic_fixture()
    original = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=fixture.query_reference_mask,
        emit_confidence_margin=True,
    )
    flipped_reference = np.asarray([[[[[0, 1]]]]], dtype=np.uint8)
    altered = run_phase5_comparison(
        query_feature_encoding=fixture.query_feature_encoding,
        support_records=fixture.support_records,
        query_patient_id=fixture.query_patient_id,
        query_case_id=fixture.query_case_id,
        query_identity=fixture.query_identity,
        dataset_manifest_hash=fixture.dataset_manifest_hash,
        query_reference_mask=flipped_reference,
        emit_confidence_margin=True,
    )

    for first, second in zip(original.records, altered.records, strict=True):
        assert first.retrieval_result_identity_sha256 == second.retrieval_result_identity_sha256
        assert (
            first.foreground_prototype_identity_sha256
            == second.foreground_prototype_identity_sha256
        )
        assert (
            first.background_prototype_identity_sha256
            == second.background_prototype_identity_sha256
        )
        assert first.inference_identity_sha256 == second.inference_identity_sha256
        assert first.prediction_content_sha256 == second.prediction_content_sha256
        assert (first.dice, first.iou) != (second.dice, second.iou)


def test_patient_leakage_rejected() -> None:
    fixture = build_phase5_synthetic_fixture()
    first_support = fixture.support_records[0]
    leaked_supports = (
        type(first_support)(
            schema_name=first_support.schema_name,
            schema_version=first_support.schema_version,
            support_identifier=first_support.support_identifier,
            support_patient_id=fixture.query_patient_id,
            support_case_id=first_support.support_case_id,
            support_manifest_hash=first_support.support_manifest_hash,
            dataset_manifest_hash=first_support.dataset_manifest_hash,
            feature_encoding=first_support.feature_encoding,
            binary_mask=first_support.binary_mask,
        ),
        fixture.support_records[1],
    )

    with pytest.raises(Phase5ComparisonLeakageError):
        run_phase5_comparison(
            query_feature_encoding=fixture.query_feature_encoding,
            support_records=leaked_supports,
            query_patient_id=fixture.query_patient_id,
            query_case_id=fixture.query_case_id,
            query_identity=fixture.query_identity,
            dataset_manifest_hash=fixture.dataset_manifest_hash,
            query_reference_mask=fixture.query_reference_mask,
            emit_confidence_margin=True,
        )


def test_case_leakage_rejected() -> None:
    fixture = build_phase5_synthetic_fixture()
    first_support = fixture.support_records[0]
    leaked_supports = (
        type(first_support)(
            schema_name=first_support.schema_name,
            schema_version=first_support.schema_version,
            support_identifier=first_support.support_identifier,
            support_patient_id=first_support.support_patient_id,
            support_case_id=fixture.query_case_id,
            support_manifest_hash=first_support.support_manifest_hash,
            dataset_manifest_hash=first_support.dataset_manifest_hash,
            feature_encoding=first_support.feature_encoding,
            binary_mask=first_support.binary_mask,
        ),
        fixture.support_records[1],
    )

    with pytest.raises(Phase5ComparisonLeakageError):
        run_phase5_comparison(
            query_feature_encoding=fixture.query_feature_encoding,
            support_records=leaked_supports,
            query_patient_id=fixture.query_patient_id,
            query_case_id=fixture.query_case_id,
            query_identity=fixture.query_identity,
            dataset_manifest_hash=fixture.dataset_manifest_hash,
            query_reference_mask=fixture.query_reference_mask,
            emit_confidence_margin=True,
        )


def test_duplicate_support_identifier_rejected() -> None:
    fixture = build_phase5_synthetic_fixture()
    first = fixture.support_records[0]
    duplicated = (
        first,
        type(first)(
            schema_name=fixture.support_records[1].schema_name,
            schema_version=fixture.support_records[1].schema_version,
            support_identifier=first.support_identifier,
            support_patient_id=fixture.support_records[1].support_patient_id,
            support_case_id=fixture.support_records[1].support_case_id,
            support_manifest_hash=fixture.support_records[1].support_manifest_hash,
            dataset_manifest_hash=fixture.support_records[1].dataset_manifest_hash,
            feature_encoding=fixture.support_records[1].feature_encoding,
            binary_mask=fixture.support_records[1].binary_mask,
        ),
    )

    with pytest.raises(Phase5ComparisonValidationError):
        run_phase5_comparison(
            query_feature_encoding=fixture.query_feature_encoding,
            support_records=duplicated,
            query_patient_id=fixture.query_patient_id,
            query_case_id=fixture.query_case_id,
            query_identity=fixture.query_identity,
            dataset_manifest_hash=fixture.dataset_manifest_hash,
            query_reference_mask=fixture.query_reference_mask,
            emit_confidence_margin=True,
        )
