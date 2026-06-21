from __future__ import annotations

from ..office.types import Modality
from .base import Extractor
from .base import ExtractResult


class TextExtractor(Extractor):
    modality = Modality.TEXT

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 BOM
            text = data.decode("utf-16")
        else:
            for enc in ("utf-8-sig", "utf-8"):
                try:
                    text = data.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = data.decode("latin-1")  # never raises; final fallback
        return ExtractResult(segments=[text], modality=self.modality)
