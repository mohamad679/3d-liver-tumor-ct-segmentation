"""Synthetic Phase 1 report generation for ProtoEM-CT."""

from protoem_ct.reporting.generate import (
    REPORT_FORMAT,
    REPORT_NUMERIC_PRECISION,
    REPORT_STAGE,
    ReportGenerationError,
    ReportInputArtifactError,
    ReportOutputCollisionError,
    ReportPathError,
    ReportRenderingError,
    generate_synthetic_report,
)

__all__ = [
    "REPORT_FORMAT",
    "REPORT_NUMERIC_PRECISION",
    "REPORT_STAGE",
    "ReportGenerationError",
    "ReportInputArtifactError",
    "ReportOutputCollisionError",
    "ReportPathError",
    "ReportRenderingError",
    "generate_synthetic_report",
]
