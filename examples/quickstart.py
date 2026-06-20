"""
Set OPENAI_API_KEY and MISTRAL_API_KEY in the environment, drop a few files
(pdf / docx / image / audio / text) into ./docs, then run this.
"""
from __future__ import annotations

from pathlib import Path

from bicardinal import Bicardinal, Config, ExtractionError


def show(name: str, stage: str, done: int, total: int) -> None:
    end = "\r" if done < total else "\n"
    print(f"  [{name}] {stage:<8} {done}/{total}", end=end)


def main() -> None:
    docs = sorted(Path("./docs").glob("*"))
    if not docs:
        print("put some files in ./docs first")
        return

    store = Bicardinal("./bicardinal_data", config=Config(chunk_size=512, overlap=0.1))

    if "demo" in store.list():
        store.delete("demo")
    col = store.create("demo")

    col.init("insert")
    for path in docs:
        try:
            result = col.ingest(
                path.name,
                path.read_bytes(),
                on_progress=lambda s, d, t, n=path.name: show(n, s, d, t),
            )
        except ExtractionError as e:
            print(f"  skip {path.name}: {e}")
            continue
        note = f" ({len(result.errors)} chunk errors)" if result.errors else ""
        print(f"  {path.name}: {result.n_chunks} chunks{note}")
    failed = col.finalize()
    if failed:
        print("write-failed documents:", failed)

    st = col.status()
    print(f"\ncollection 'demo': {st.n_files} files, {st.n_chunks} chunks")

    query = "what is this about?"

    print(f"\n# search {query!r}")
    for h in col.search(query, k=5):
        print(f"  {h.score:.3f}  {h.filename}#{h.chunk_index}  {h.raw_text[:70]!r}")

    print(f"\n# most_similar_files {query!r}")
    for fh in col.most_similar_files(query, k=3):
        print(f"  {fh.score:.3f}  {fh.filename}")

    target = st.filenames[0]
    print(f"\n# search_in_file {target!r} {query!r}")
    for h in col.search_in_file(query, target, k=5):
        print(f"  {h.score:.3f}  #{h.chunk_index}  {h.raw_text[:70]!r}")

    col.close()


if __name__ == "__main__":
    main()
