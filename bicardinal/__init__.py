from __future__ import annotations

import shutil
from pathlib import Path

from mistralai import Mistral
from openai import OpenAI

from .extractors.audio import AudioExtractor
from .extractors.docx import DocxExtractor
from .extractors.image import ImageExtractor
from .extractors.pdf import PdfExtractor
from .extractors.router import Router
from .extractors.text import TextExtractor
from .office.config import Config
from .services.embedder import Embedder
from .services.summarizer import Summarizer

from .office.collection import Collection, CollectionStatus
from .office.types import AddResult, FileHit, Modality, SearchHit, Usage
from .office.exceptions import (
    BicardinalError,
    CollectionExists,
    CollectionNotFound,
    DocumentNotFound,
    DuplicateDocument,
    EmptyFile,
    ExtractionError,
    UnsupportedFileType,
)
import re


class Bicardinal:
    """Owns a directory of collections and the shared models/clients they use."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        config: Config | None = None,
        openai_api_key: str | None = None,
        mistral_api_key: str | None = None,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._config = config or Config()

        openai_client = OpenAI(api_key=openai_api_key)    # shared across summarizer/image/audio
        mistral_client = Mistral(api_key=mistral_api_key)

        self._embedder = Embedder(
            self._config.embed_model,
            batch_size=self._config.embed_batch_size,
            device=self._config.embed_device,
            doc_prompt=self._config.embed_doc_prompt,
            query_prompt=self._config.embed_query_prompt,
        )
        self._summarizer = Summarizer(
            openai_client,
            self._config.summarizer_model,
            max_concurrency=self._config.summarizer_max_concurrency,
            reasoning_effort=self._config.summarizer_reasoning_effort,
        )
        self._router = Router(
            {
                Modality.TEXT: TextExtractor(),
                Modality.DOCX: DocxExtractor(),
                Modality.PDF: PdfExtractor(mistral_client, self._config.ocr_model),
                Modality.IMAGE: ImageExtractor(openai_client, self._config.image_model),
                Modality.AUDIO: AudioExtractor(openai_client, self._config.transcribe_model),
            }
        )

    def _collection_path(self, name: str) -> Path:
        if not name.lower().replace("_", "").isalnum():
            raise ValueError(f"invalid collection name. Only [A-z0-9_] are supported: {name!r}")
        return self._root / name

    def _build_collection(self, path: Path) -> Collection:
        return Collection(
            path,
            config=self._config,
            embedder=self._embedder,
            summarizer=self._summarizer,
            router=self._router,
        )

    def _is_collection(self, path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())  # exists, and has store files

    def create(self, name: str) -> Collection:
        path = self._collection_path(name)
        if path.exists():
            raise CollectionExists(name)
        return self._build_collection(path)  # Collection.__init__ mkdirs + inits empty stores

    def open(self, name: str) -> Collection:
        path = self._collection_path(name)
        if not self._is_collection(path):
            raise CollectionNotFound(name)
        return self._build_collection(path)

    def list(self) -> list[str]:
        return sorted(p.name for p in self._root.iterdir() if self._is_collection(p))

    def delete(self, name: str) -> None:
        path = self._collection_path(name)
        if not self._is_collection(path):
            raise CollectionNotFound(name)
        col = self._build_collection(path)
        col.destroy()  # clear brinicle stores
        shutil.rmtree(path, ignore_errors=True)  # remove the now-empty directory

    def destroy(self) -> None:
        for name in self.list():  # snapshot first, then mutate the filesystem
            self.delete(name)


__all__ = [
    "Bicardinal",
    "Collection",
    "Config",
    "CollectionStatus",
    "AddResult",
    "SearchHit",
    "FileHit",
    "Usage",
    "Modality",
    "BicardinalError",
    "DuplicateDocument",
    "DocumentNotFound",
    "CollectionExists",
    "CollectionNotFound",
    "UnsupportedFileType",
    "EmptyFile",
    "ExtractionError",
]