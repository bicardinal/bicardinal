
# bicardinal

**Multimodal retrieval engine.** Drop in PDFs, Word docs, images, audio, or
plain text, bicardinal extracts, summarizes, embeds, and indexes them so you
can search across everything with one call.

## Install

```bash
pip install bicardinal
```

Set your provider keys (used for OCR, vision, transcription, and summaries):

```bash
export OPENAI_API_KEY=sk-...
export MISTRAL_API_KEY=...
```

## Deadly simple

```python
from pathlib import Path
from bicardinal import Bicardinal

store = Bicardinal("./data")          # a home for your collections
col = store.create("docs")            # make a collection
col.init("build")                     # open it for ingestion

# Throw any file at it, pdf, docx, png, mp3, txt...
for path in Path("./my_files").glob("*"):
    col.ingest(path.name, path.read_bytes())

col.finalize() # build the index

# Search across everything
for hit in col.search("quarterly revenue growth", k=5):
    print(f"{hit.score:.3f}  {hit.filename}#{hit.chunk_index}  {hit.raw_text[:80]}")

col.close()
```

That's it. No pipelines to wire, no embedding code to write.

## A little more

```python
# Rank whole files by relevance
for f in col.most_similar_files("budget forecast", k=3):
    print(f.score, f.filename)

# Search within a single document
col.search_in_file("conclusion", "report.pdf", k=5)

# Reopen later, it's all on disk
col = store.open("docs")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a fuller tour.

## License

Apache-2.0
