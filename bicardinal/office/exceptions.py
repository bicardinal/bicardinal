from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Usage


class BicardinalError(Exception):
    """Base class for all bicardinal errors."""


class DuplicateDocument(BicardinalError):
    """A document with this filename already exists in the collection."""


class DocumentNotFound(BicardinalError):
    """No document with this filename exists in the collection."""


class UnsupportedFileType(BicardinalError):
    """Detected MIME type has no registered extractor."""


class EmptyFile(BicardinalError):
    """Input was zero bytes."""


class ExtractionError(BicardinalError):
    """An extractor (OCR, transcription, decode, ...) failed.

    ``usage`` holds whatever was already billed before the failure (OCR
    batches or audio pieces that succeeded), so the spend is not lost.
    """

    def __init__(self, message: str, *, usage: "Usage | None" = None) -> None:
        super().__init__(message)
        if usage is None:
            from .types import Usage

            usage = Usage()
        self.usage = usage


class CollectionExists(BicardinalError):
    """A collection with this name already exists."""


class CollectionNotFound(BicardinalError):
    """No collection with this name exists."""
