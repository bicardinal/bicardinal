from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import brinicle

from ..office.types import ChunkRecord


class PayloadStore:
    """Thin wrapper over brinicle.PayloadStore, ChunkRecord in, ChunkRecord out."""

    def __init__(self, path: str | Path, *, shard_count: int = 4) -> None:
        self._store = brinicle.PayloadStore().init(str(path), shard_count=shard_count)

    @staticmethod
    def _encode(records: dict[str, ChunkRecord]) -> tuple[list[str], list[str]]:
        ids = list(records.keys())
        values = [json.dumps(asdict(r)) for r in records.values()]
        return ids, values

    def insert(self, records: dict[str, ChunkRecord]) -> None:
        ids, values = self._encode(records)
        self._store.insert(ids, values)

    def upsert(self, records: dict[str, ChunkRecord]) -> None:
        ids, values = self._encode(records)
        self._store.upsert(ids, values)

    def get(self, chunk_ids: list[str]) -> list[ChunkRecord | None]:
        values = self._store.retrieve(chunk_ids)
        return [ChunkRecord(**json.loads(v)) if v is not None else None for v in values]

    def delete(self, chunk_ids: list[str]) -> int:
        return self._store.delete(chunk_ids)

    def close(self):
        return self._store.close()

    def destroy(self):
        return self._store.destroy()
