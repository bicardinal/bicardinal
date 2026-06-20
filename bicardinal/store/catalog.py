from __future__ import annotations
from pathlib import Path
import json
import brinicle

FILES_KEY = "$__files__$"


class Catalog:
    def __init__(self, path: str | Path, *, shard_count: int = 4) -> None:
        self._store = brinicle.PayloadStore().init(str(path), shard_count=shard_count)

    def get_ids(self, filename: str) -> list[str]:
        v = self._store.retrieve([filename])[0]
        return json.loads(v) if v is not None else []

    def get_files(self) -> list[str]:
        v = self._store.retrieve([FILES_KEY])[0]
        return json.loads(v) if v is not None else []

    def add_ids(self, filename: str, chunk_ids: list[str]) -> None:
        assert filename != FILES_KEY, "reserved filename"
        ids = self.get_ids(filename)
        ids.extend(chunk_ids)
        self._store.upsert([filename], [json.dumps(ids)])
        files = self.get_files()
        if filename not in files:  # register a new file in the enumeration index
            files.append(filename)
            self._store.upsert([FILES_KEY], [json.dumps(files)])

    def delete_file(self, filename: str) -> list[str]:
        ids = self.get_ids(filename)
        self._store.delete([filename])
        files = self.get_files()
        if filename in files:
            files.remove(filename)
            self._store.upsert([FILES_KEY], [json.dumps(files)])
        return ids  # so caller can purge index + payload

    def delete_ids(self, filename: str, chunk_ids: list[str]) -> None:
        remove = set(chunk_ids)
        ids = [i for i in self.get_ids(filename) if i not in remove]
        self._store.upsert([filename], [json.dumps(ids)])

    def has_file(self, filename: str) -> bool:
        return filename in self.get_files()

    def close(self) -> None:
        self._store.close()

    def destroy(self) -> None:
        self._store.destroy()