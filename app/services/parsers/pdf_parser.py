from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models import FileDescriptor, ParsedDocument, ParsedSection
from app.services.parsers.assets import write_image_asset
from app.services.parsers.base import BaseParser

logger = get_logger(component="pdf_parser")

# Skip tiny raster images (bullets, logos, separators) when extracting visuals.
_MIN_IMAGE_DIMENSION = 48


class PDFParser(BaseParser):
    def parse(self, file_descriptor: FileDescriptor) -> ParsedDocument:
        page_texts, page_tables = self._extract_text_and_tables(file_descriptor)
        page_images = self._extract_page_images(file_descriptor)

        sections: list[ParsedSection] = []
        all_pages = sorted(set(page_texts) | set(page_tables) | set(page_images))

        for page_number in all_pages:
            visual_refs = page_images.get(page_number, [])

            text = page_texts.get(page_number, "").strip()
            if text:
                sections.append(
                    ParsedSection(
                        text=text,
                        page_number=page_number,
                        metadata={"visual_refs": visual_refs},
                    )
                )
            elif visual_refs:
                # Page has images but no extractable text — still surface the visuals.
                summary = "\n".join(f"[IMAGE] name={ref['name']}" for ref in visual_refs)
                sections.append(
                    ParsedSection(
                        text=summary,
                        page_number=page_number,
                        section_title=f"Page {page_number} visuals",
                        metadata={"visual_refs": visual_refs},
                    )
                )

            for table_index, table_text in enumerate(page_tables.get(page_number, []), start=1):
                sections.append(
                    ParsedSection(
                        text=table_text,
                        page_number=page_number,
                        section_title=f"Page {page_number} table {table_index}",
                    )
                )

        return ParsedDocument(document_id=file_descriptor.document_id, sections=sections)

    @classmethod
    def _extract_text_and_tables(
        cls, file_descriptor: FileDescriptor
    ) -> tuple[dict[int, str], dict[int, list[str]]]:
        texts: dict[int, str] = {}
        tables: dict[int, list[str]] = {}
        try:
            import pdfplumber

            with pdfplumber.open(str(file_descriptor.file_path)) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        texts[index] = text

                    page_tables: list[str] = []
                    for table in page.extract_tables() or []:
                        formatted = cls._format_table(table)
                        if formatted:
                            page_tables.append(formatted)
                    if page_tables:
                        tables[index] = page_tables
        except Exception as error:
            logger.warning(
                "pdfplumber extraction failed, falling back to PyMuPDF for text",
                file_path=str(file_descriptor.file_path),
                error=str(error),
            )

        if not texts:
            import fitz

            doc = fitz.open(str(file_descriptor.file_path))
            try:
                for index, page in enumerate(doc, start=1):
                    text = (page.get_text("text") or "").strip()
                    if text:
                        texts[index] = text
            finally:
                doc.close()
        return texts, tables

    @staticmethod
    def _format_table(table) -> str | None:
        row_lines = []
        for row in table:
            cells = [(cell or "").strip().replace("\n", " ") for cell in row]
            if any(cells):
                row_lines.append(" | ".join(cells))
        return "[TABLE]\n" + "\n".join(row_lines) if row_lines else None

    @staticmethod
    def _extract_page_images(file_descriptor: FileDescriptor) -> dict[int, list[dict[str, str]]]:
        images: dict[int, list[dict[str, str]]] = {}
        try:
            import fitz

            doc = fitz.open(str(file_descriptor.file_path))
            try:
                for page_index, page in enumerate(doc, start=1):
                    refs: list[dict[str, str]] = []
                    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                        xref = image_info[0]
                        try:
                            pixmap = fitz.Pixmap(doc, xref)
                            if pixmap.width < _MIN_IMAGE_DIMENSION or pixmap.height < _MIN_IMAGE_DIMENSION:
                                continue
                            # Normalise CMYK / alpha to RGB PNG for browser display.
                            if pixmap.n >= 5 or pixmap.alpha:
                                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                            blob = pixmap.tobytes("png")
                        except Exception:
                            continue

                        name = f"page_{page_index}_img_{image_index}"
                        path = write_image_asset(file_descriptor.file_path, blob, "png", name)
                        refs.append(
                            {
                                "type": "image",
                                "name": name,
                                "path": path,
                                "page_number": str(page_index),
                            }
                        )
                    if refs:
                        images[page_index] = refs
            finally:
                doc.close()
        except Exception as error:
            logger.warning(
                "PDF image extraction failed",
                file_path=str(file_descriptor.file_path),
                error=str(error),
            )
        return images
