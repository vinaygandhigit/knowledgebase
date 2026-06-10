from __future__ import annotations

import re

from app.domain.models import FileDescriptor, ParsedDocument, ParsedSection
from app.services.parsers.base import BaseParser
from app.services.parsers.text_like_parser import read_text_with_fallback

_IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")


class MarkdownParser(BaseParser):
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        text = read_text_with_fallback(file_descriptor)
        if not text.strip():
            return ParsedDocument(document_id=file_descriptor.document_id, sections=[])

        base_dir = file_descriptor.file_path.parent
        sections: list[ParsedSection] = []
        current_title = ""
        buffer: list[str] = []
        current_visuals: list[dict[str, str]] = []

        def flush() -> None:
            if not buffer:
                return
            section_text = "\n".join(buffer).strip()
            if section_text:
                sections.append(
                    ParsedSection(
                        text=section_text,
                        section_title=current_title,
                        metadata={"visual_refs": list(current_visuals)},
                    )
                )
            buffer.clear()
            current_visuals.clear()

        for line in text.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                flush()
                current_title = heading_match.group(2).strip()
                buffer.append(line.rstrip())
                continue

            image_match = _IMAGE_PATTERN.search(line)
            if image_match:
                alt_text = image_match.group(1).strip()
                source = image_match.group(2).strip().split()[0].strip("<>") if image_match.group(2).strip() else ""
                buffer.append(line.rstrip())
                if source:
                    buffer.append(f"[IMAGE] alt={alt_text} source={source}".strip())
                    current_visuals.append(self._build_visual_ref(base_dir, alt_text, source))
                continue

            buffer.append(line.rstrip())

        flush()
        return ParsedDocument(document_id=file_descriptor.document_id, sections=sections)

    @staticmethod
    def _build_visual_ref(base_dir, alt_text: str, source: str) -> dict[str, str]:
        name = alt_text or "image"
        if source.startswith(("http://", "https://")):
            return {"type": "image", "name": name, "url": source}

        candidate = (base_dir / source).resolve()
        if candidate.exists() and candidate.is_file():
            return {"type": "image", "name": name, "path": str(candidate)}

        # Unresolved local reference — record it so it still appears in context.
        return {"type": "image", "name": name, "source": source}
