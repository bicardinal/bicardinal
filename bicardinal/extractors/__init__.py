"""Modality extractors and the MIME router. Each extractor: bytes -> raw text segments."""
from .base import Extractor, ExtractResult
from .router import Router

__all__ = ["Extractor", "ExtractResult", "Router"]
