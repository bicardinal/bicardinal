from __future__ import annotations
import numpy as np

class Embedder:
    """sentence-transformers wrapper"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        normalize: bool = True,
        batch_size: int = 64,
        device: str | None = None,
        doc_prompt: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._batch_size = batch_size
        self._doc_prompt = doc_prompt
        self._query_prompt = query_prompt
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            prompt=self._doc_prompt,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            prompt=self._query_prompt,
            convert_to_numpy=True,
        ).astype(np.float32)

