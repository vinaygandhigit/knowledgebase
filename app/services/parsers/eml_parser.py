from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser

from app.domain.models import FileDescriptor, ParsedDocument, ParsedSection
from app.services.parsers.base import BaseParser


class EMLParser(BaseParser):
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        raw_bytes = file_descriptor.file_path.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)

        subject = message.get("subject", "")
        sender = message.get("from", "")
        date = message.get("date", "")
        body = self._extract_body(message)

        header_block = "\n".join(
            [
                f"Subject: {subject}" if subject else "",
                f"From: {sender}" if sender else "",
                f"Date: {date}" if date else "",
            ]
        ).strip()

        combined = "\n\n".join(part for part in [header_block, body] if part).strip()
        section = ParsedSection(text=combined) if combined else None
        sections = [section] if section else []

        return ParsedDocument(document_id=file_descriptor.document_id, sections=sections)

    def _extract_body(self, message) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = (part.get("Content-Disposition") or "").lower()
                if "attachment" in content_disposition:
                    continue

                payload = part.get_payload(decode=True)
                if payload is None:
                    continue

                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="ignore").strip()
                if not text:
                    continue

                if content_type == "text/plain":
                    plain_parts.append(text)
                elif content_type == "text/html":
                    html_parts.append(text)
        else:
            payload = message.get_payload(decode=True)
            if payload:
                charset = message.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="ignore").strip()
                if message.get_content_type() == "text/html":
                    html_parts.append(text)
                else:
                    plain_parts.append(text)

        if plain_parts:
            return "\n\n".join(plain_parts).strip()

        if html_parts:
            stripped = re.sub(r"<[^>]+>", " ", "\n\n".join(html_parts))
            return re.sub(r"\s+", " ", stripped).strip()

        return ""
