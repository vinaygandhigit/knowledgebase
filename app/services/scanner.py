from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.security import is_supported_extension, is_within_root, validate_max_file_size
from app.domain.models import FileDescriptor


logger = get_logger(component="file_scanner")


class FileScanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def scan(self) -> list[FileDescriptor]:
        root = self.settings.root_path.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Root path does not exist or is not a directory: {root}")

        descriptors: list[FileDescriptor] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            descriptor = self._build_descriptor(path, root)
            if descriptor:
                descriptors.append(descriptor)

        return descriptors

    def scan_single(self, file_path: Path) -> FileDescriptor:
        root = self.settings.root_path.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Root path does not exist or is not a directory: {root}")

        resolved_path = file_path.resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"File does not exist or is not a file: {resolved_path}")

        descriptor = self._build_descriptor(resolved_path, root)
        if descriptor is None:
            raise ValueError(f"File is not eligible for ingestion: {resolved_path}")
        return descriptor

    def _build_descriptor(self, path: Path, root: Path) -> FileDescriptor | None:
        if path.name.lower() == "readme.md":
            logger.info("Skipped README file", file_path=str(path))
            return None
        if not is_within_root(root, path):
            logger.warning("Skipped file outside root", file_path=str(path))
            return None
        if not is_supported_extension(path, self.settings.supported_extensions):
            return None

        stat = path.stat()
        if not validate_max_file_size(stat.st_size, self.settings.max_file_size_mb):
            logger.warning(
                "Skipped oversized file",
                file_path=str(path),
                file_size=stat.st_size,
                max_mb=self.settings.max_file_size_mb,
            )
            return None

        checksum = self._sha256(path)
        return FileDescriptor(
            document_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()))),
            file_name=path.name,
            file_path=path.resolve(),
            file_extension=path.suffix.lower(),
            file_size=stat.st_size,
            created_timestamp=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_timestamp=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum_sha256=checksum,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        hash_sha = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hash_sha.update(block)
        return hash_sha.hexdigest()
