from __future__ import annotations

from io import BytesIO

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..office.types import Modality
from .base import Extractor
from .base import ExtractResult


def _iter_blocks(doc):
    # walk body children in document order so tables stay where they belong
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


class DocxExtractor(Extractor):
    modality = Modality.DOCX

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        doc = Document(BytesIO(data))
        parts: list[str] = []
        for block in _iter_blocks(doc):
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text)
            else:  # Table
                for row in block.rows:
                    line = "\t".join(c.text.strip() for c in row.cells).strip()
                    if line:
                        parts.append(line)
        return ExtractResult(segments=["\n".join(parts)], modality=self.modality)
