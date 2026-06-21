from __future__ import annotations

import math
from pathlib import Path

import brinicle
import numpy as np


def _build_lexical_config() -> brinicle.LexicalConfig:
    cfg = brinicle.LexicalConfig()
    cfg.build_title_weight = 0.0
    cfg.search_title_weight = 0.0
    cfg.build_attr_weight = 0.0
    cfg.search_attr_weight = 0.0
    cfg.build_subcategory_weight = 0.0
    cfg.search_subcategory_weight = 0.0
    cfg.build_category_weight = 1.0
    cfg.search_category_weight = 1.0
    cfg.build_vector_weight = 1.0
    cfg.search_vector_weight = 1.0
    cfg.build_category_penalty = 1.0
    cfg.search_category_penalty = 1e8
    cfg.vector_normalized = True
    return cfg


class Index:
    def __init__(
        self,
        path: str | Path,
        vector_dim: int,
        *,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        n_shards: int = 1,
        build_n_threads: int = 1,
    ) -> None:
        self._vector_dim = vector_dim
        self._engine = brinicle.ItemSearchEngine(
            str(path),
            dim=7,
            vector_dim=vector_dim,
            lexical_config=_build_lexical_config(),
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            n_shards=n_shards,
            build_n_threads=build_n_threads,
            title_ratio=1.0,
        )

    def init(self, mode: str = "build") -> None:
        self._engine.init(mode)

    def ingest(self, chunk_id: str, vector: np.ndarray, filename: str) -> None:
        assert vector.shape == (self._vector_dim,), "vector dim mismatch"
        self._engine.ingest(chunk_id, "", category=filename, vector=vector)

    def finalize(self) -> None:
        self._engine.finalize()

    def search_with_distance(
        self,
        vector: np.ndarray,
        k: int = 10,
        *,
        category: str | None = None,
        threshold: float = math.inf,
        efs: int | None = None,
        n_jobs: int = 1,
    ) -> list[tuple[str, float]]:
        return self._engine.search_with_distance(
            "",
            k=k,
            efs=efs,
            threshold=threshold,
            category=category,
            vector=vector,
            n_jobs=n_jobs,
        )

    def delete(self, chunk_ids: list[str]) -> None:
        self._engine.delete_items(chunk_ids)

    def close(self) -> None:
        self._engine.close()

    def destroy(self) -> None:
        self._engine.destroy()
