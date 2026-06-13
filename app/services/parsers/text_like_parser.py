from __future__ import annotations

from app.domain.models import FileDescriptor, ParsedDocument, ParsedSection
from app.services.parsers.base import BaseParser


def read_text_with_fallback(file_descriptor: FileDescriptor) -> str:
    try:
        return file_descriptor.file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_descriptor.file_path.read_bytes()
        import chardet

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        return raw.decode(encoding=encoding, errors="ignore")


class TextLikeParser(BaseParser):
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        text = read_text_with_fallback(file_descriptor)
        section = ParsedSection(text=text.strip()) if text.strip() else None
        sections = [section] if section else []
        return ParsedDocument(document_id=file_descriptor.document_id, sections=sections)
