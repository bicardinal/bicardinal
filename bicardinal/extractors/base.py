from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from ..office.types import Modality
from ..office.types import Usage


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
