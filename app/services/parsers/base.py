from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import FileDescriptor, ParsedDocument


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        raise NotImplementedError
