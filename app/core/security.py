from __future__ import annotations

from pathlib import Path


class SecurityError(Exception):
    pass


def is_supported_extension(file_path: Path, supported_extensions: list[str]) -> bool:
    return file_path.suffix.lower() in supported_extensions


def is_within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_max_file_size(file_size_bytes: int, max_size_mb: int) -> bool:
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size_bytes <= max_size_bytes
