"""Canonical URI-aware filesystem-path handling for persisted Phase 3 logs."""

from __future__ import annotations

import re

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?:^|(?<=[\s\"']))[A-Za-z]:[\\/]")
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![:A-Za-z0-9_+.-])/(?!/)[^\s'\"`<>|]+")
_QUOTED_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?P<quote>[\"'])(?P<path>/(?!/)[^\"'\r\n]+)(?P=quote)"
)
_APPROVED_REFERENCE_RE = re.compile(
    r"(?:\b(?:https?|mlflow-artifacts):/[^\s'\"`<>|]*|"
    r"<(?:ABSOLUTE_PATH|BASELINE_ENV|HOME|NNUNET_PREPROCESSED|NNUNET_RAW|"
    r"NNUNET_RESULTS|REPOSITORY_ROOT|RUNTIME_ROOT|TEMP_ROOT)>"
    r"(?:/[^\s'\"`<>|]+)*)",
    flags=re.IGNORECASE,
)
_TRAILING_PATH_PUNCTUATION = ",.;!?)]}"


def _redact_residual_absolute_paths(value: str) -> str:
    """Redact filesystem paths while preserving approved non-file URIs."""

    approved_uris: list[str] = []

    def protect_uri(match: re.Match[str]) -> str:
        approved_uris.append(match.group(0))
        return f"<APPROVED_URI_{len(approved_uris) - 1}>"

    protected = _APPROVED_REFERENCE_RE.sub(protect_uri, value)
    protected = _QUOTED_POSIX_ABSOLUTE_PATH_RE.sub(
        lambda match: f"{match.group('quote')}<ABSOLUTE_PATH>{match.group('quote')}",
        protected,
    )
    protected = _POSIX_ABSOLUTE_PATH_RE.sub(_replace_unquoted_absolute_path, protected)
    for index, uri in enumerate(approved_uris):
        protected = protected.replace(f"<APPROVED_URI_{index}>", uri)
    return protected


def _contains_absolute_filesystem_path(value: str) -> bool:
    """Return whether text contains a local absolute path outside approved URI schemes."""

    protected = _APPROVED_REFERENCE_RE.sub("<APPROVED_REFERENCE>", value)
    if _QUOTED_POSIX_ABSOLUTE_PATH_RE.search(protected) is not None:
        return True
    if _POSIX_ABSOLUTE_PATH_RE.search(protected) is not None:
        return True
    return _WINDOWS_ABSOLUTE_PATH_RE.search(protected) is not None


def _replace_unquoted_absolute_path(match: re.Match[str]) -> str:
    token = match.group(0)
    punctuation = ""
    while token and token[-1] in _TRAILING_PATH_PUNCTUATION:
        punctuation = token[-1] + punctuation
        token = token[:-1]
    return f"<ABSOLUTE_PATH>{punctuation}"
