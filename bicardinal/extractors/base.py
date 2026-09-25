from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from ..office.types import Modality
from ..office.types import Usage


@dataclass
class ExtractBatch:
    """One piece of an extraction that runs in pieces: which piece, of how
    many, the text it produced, and what it cost on its own."""

    index: int
    total: int
    segments: list[str]
    usage: Usage = field(default_factory=Usage)
    # An image's description, written by vision alongside its text; None for
    # everything that is described later from its chunks.
    descriptions: list[str] | None = None


@dataclass
class ExtractResult:
    segments: list[str]
    modality: Modality
    prebuilt_descriptions: list[str] | None = None
    usage: Usage = field(default_factory=Usage)


class Extractor:
    """Base class for modality extractors: bytes -> ExtractResult."""

    modality: Modality

    def extract(self, data: bytes, *, filename: str | None = None) -> ExtractResult:
        raise NotImplementedError
