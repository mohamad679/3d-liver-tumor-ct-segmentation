"""Internal JSON publication helpers for Phase 2 data artifacts."""

from __future__ import annotations

import os
from pathlib import Path


class Phase2PublicationExistingOutputError(RuntimeError):
    """Raised when a final or temporary Phase 2 publication target already exists."""


class Phase2PublicationIOError(RuntimeError):
    """Raised when a Phase 2 publication write, link, rename, or cleanup operation fails."""


def publish_text_no_overwrite(
    *,
    text: str,
    output_path: Path,
    temporary_exists_message: str,
    final_exists_message: str,
) -> None:
    """Publish UTF-8 text beside the final output without intentionally overwriting artifacts.

    The preferred path writes a temporary file in the final output directory, then hard-links that
    temp file to the final output. On filesystems that support hard links, this gives an atomic
    no-overwrite final publication because creating the final hard link fails if the final path
    already exists.

    Some external filesystems do not support hard links. The fallback keeps the temp file in the
    final output directory, rechecks that the final output does not exist, then renames the temp
    file to the final path. Python's standard library has no portable "rename only if destination
    does not exist" operation, so this fallback never intentionally overwrites a pre-existing
    artifact but cannot eliminate the narrow race where another process creates the final path
    between the existence check and `os.rename`.
    """
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    created_temp = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            raise Phase2PublicationExistingOutputError(temporary_exists_message)
        temp_path.write_text(text, encoding="utf-8", newline="\n")
        created_temp = True
        if output_path.exists():
            raise Phase2PublicationExistingOutputError(final_exists_message)
        try:
            os.link(temp_path, output_path)
        except OSError as link_exc:
            if output_path.exists():
                raise Phase2PublicationExistingOutputError(final_exists_message) from link_exc
            os.rename(temp_path, output_path)
        else:
            temp_path.unlink()
        created_temp = False
    except Phase2PublicationExistingOutputError:
        raise
    except OSError as exc:
        raise Phase2PublicationIOError("failed to publish Phase 2 JSON artifact") from exc
    finally:
        if created_temp and temp_path.exists():
            temp_path.unlink()
