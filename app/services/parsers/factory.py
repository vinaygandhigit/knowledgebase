from __future__ import annotations

from app.services.parsers.base import BaseParser
from app.services.parsers.docx_parser import DOCXParser
from app.services.parsers.eml_parser import EMLParser
from app.services.parsers.markdown_parser import MarkdownParser
from app.services.parsers.pdf_parser import PDFParser
from app.services.parsers.pptx_parser import PPTXParser
from app.services.parsers.text_like_parser import TextLikeParser
from app.services.parsers.txt_parser import TXTParser
from app.services.parsers.xlsx_parser import XLSXParser


class ParserFactory:
    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".pptx": PPTXParser(),
            ".xlsx": XLSXParser(),
            ".txt": TXTParser(),
            ".md": MarkdownParser(),
            ".yaml": TextLikeParser(),
            ".yml": TextLikeParser(),
            ".sh": TextLikeParser(),
            ".toml": TextLikeParser(),
            ".eml": EMLParser(),
        }

    def get_parser(self, extension: str) -> BaseParser:
        parser = self._parsers.get(extension.lower())
        if not parser:
            raise ValueError(f"Unsupported extension: {extension}")
        return parser
