from __future__ import annotations

import magic

from ..office.exceptions import EmptyFile, UnsupportedFileType
from ..office.types import Modality
from .base import Extractor, ExtractResult

MIME_TO_MODALITY: dict[str, Modality] = {
    "application/pdf": Modality.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Modality.DOCX,
    "image/png": Modality.IMAGE,
    "image/jpeg": Modality.IMAGE,
    "image/webp": Modality.IMAGE,
    "image/gif": Modality.IMAGE,
    "audio/mpeg": Modality.AUDIO,
    "audio/wav": Modality.AUDIO,
    "audio/x-wav": Modality.AUDIO,
    "audio/mp4": Modality.AUDIO,
    "audio/x-m4a": Modality.AUDIO,
    "audio/ogg": Modality.AUDIO,
    "audio/flac": Modality.AUDIO,
}

MIME_TO_EXTENSION: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
}


class Router:
    def __init__(self, extractors: dict[Modality, Extractor]) -> None:
        self._extractors = extractors

    def detect(self, data: bytes) -> tuple[Modality, str, str]:
        if not data:
            raise EmptyFile("empty input")
        mime = magic.from_buffer(data, mime=True)
        modality = MIME_TO_MODALITY.get(mime)
        if modality is None and mime.startswith("text/"):
            modality = Modality.TEXT  # any text/* -> plain text extractor
        if modality is None:
            raise UnsupportedFileType(mime)
        ext = MIME_TO_EXTENSION.get(mime, ".txt" if modality is Modality.TEXT else "")
        return modality, mime, f"upload{ext}"

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        modality, mime, suggested = self.detect(data)
        extractor = self._extractors.get(modality)
        if extractor is None:
            raise UnsupportedFileType(f"no extractor for {modality.value} ({mime})")
        return extractor.extract(data, filename=filename or suggested)