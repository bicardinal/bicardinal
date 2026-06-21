from __future__ import annotations


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
    """An extractor (OCR, transcription, decode, ...) failed."""


class CollectionExists(BicardinalError):
    """A collection with this name already exists."""


class CollectionNotFound(BicardinalError):
    """No collection with this name exists."""
