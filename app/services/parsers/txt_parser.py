from __future__ import annotations

from app.domain.models import FileDescriptor, ParsedDocument, ParsedSection
from app.services.parsers.base import BaseParser
from app.services.parsers.text_like_parser import read_text_with_fallback


class TXTParser(BaseParser):
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        text = read_text_with_fallback(file_descriptor)

        section = ParsedSection(text=text.strip()) if text.strip() else None
        sections = [section] if section else []
        return ParsedDocument(document_id=file_descriptor.document_id, sections=sections)
