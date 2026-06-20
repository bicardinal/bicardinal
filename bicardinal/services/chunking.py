from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

_TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer.json"
_tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))


def chunk_text(text: str, *, chunk_size: int, overlap: float) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    ids = _tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        return []
    stride = max(1, round(chunk_size * (1.0 - overlap)))  # how far the window jumps
    chunks: list[str] = []
    start = 0
    n = len(ids)
    while start < n:
        chunks.append(_tokenizer.decode(ids[start : start + chunk_size]))
        if start + chunk_size >= n:
            break
        start += stride
    return chunks
