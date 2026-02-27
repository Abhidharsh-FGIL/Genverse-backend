"""FAISS-based vector index service for Knowledge Vault semantic search.

Each user gets their own per-user FAISS index stored as two files:
  {STORAGE_ROOT}/faiss_indexes/{user_id}.index  — the FAISS flat index
  {STORAGE_ROOT}/faiss_indexes/{user_id}.json   — chunk_id mapping list

Architecture:
  • IndexFlatIP  (inner-product on L2-normalised vectors)  ≡  cosine similarity
  • Vectors are L2-normalised before add/search so inner-product == cosine score
  • Chunk IDs (UUIDs) are kept in a sidecar JSON list at the same position as
    their FAISS row — FAISS row i  →  mapping[i]  →  DocChunk.id

Deletion / rebuild:
  • FAISS flat indexes don't support in-place removal.
  • remove_chunks() reconstructs the index omitting the deleted rows.
"""

import json
from pathlib import Path
from typing import Optional

EMBEDDING_DIM = 768  # Gemini text-embedding-004 / OpenAI text-embedding-3-small@768d


class FAISSService:
    """Per-user FAISS index manager for Knowledge Vault similarity search."""

    def __init__(self, storage_root: str) -> None:
        self._root = Path(storage_root) / "faiss_indexes"
        self._root.mkdir(parents=True, exist_ok=True)

    # ── private helpers ────────────────────────────────────────────────────

    def _idx_path(self, user_id: str) -> Path:
        return self._root / f"{user_id}.index"

    def _map_path(self, user_id: str) -> Path:
        return self._root / f"{user_id}.json"

    def _load(self, user_id: str):
        """Return (faiss_index, chunk_id_list).  Creates empty index if none exists."""
        import faiss  # lazy import — server starts even if FAISS is missing

        idx_path = self._idx_path(user_id)
        map_path = self._map_path(user_id)

        if idx_path.exists() and map_path.exists():
            index = faiss.read_index(str(idx_path))
            mapping: list[str] = json.loads(map_path.read_text())
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            mapping = []

        return index, mapping

    def _save(self, user_id: str, index, mapping: list[str]) -> None:
        import faiss
        faiss.write_index(index, str(self._idx_path(user_id)))
        self._map_path(user_id).write_text(json.dumps(mapping))

    @staticmethod
    def _to_np(vectors: list[list[float]]):
        """Convert list-of-lists to a float32 numpy matrix."""
        import numpy as np
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _normalize(mat) -> None:
        """L2-normalise rows in-place (inner-product becomes cosine similarity)."""
        import faiss
        faiss.normalize_L2(mat)

    # ── public API ─────────────────────────────────────────────────────────

    def add_batch(
        self,
        user_id: str,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Append multiple chunk embeddings to the user's FAISS index."""
        if not chunk_ids or not embeddings:
            return

        index, mapping = self._load(user_id)
        mat = self._to_np(embeddings)
        self._normalize(mat)
        index.add(mat)
        mapping.extend(chunk_ids)
        self._save(user_id, index, mapping)

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 15,
    ) -> list[str]:
        """Return up to k chunk_ids ranked by cosine similarity (highest first)."""
        index, mapping = self._load(user_id)

        if index.ntotal == 0 or not mapping:
            return []

        mat = self._to_np([query_embedding])
        self._normalize(mat)

        k = min(k, index.ntotal)
        _scores, indices = index.search(mat, k)

        return [
            mapping[i]
            for i in indices[0]
            if 0 <= i < len(mapping)
        ]

    def remove_chunks(self, user_id: str, chunk_ids: set[str]) -> None:
        """Remove specific chunks from the index by rebuilding without them.

        FAISS flat indexes don't support in-place deletion, so we reconstruct
        a new index that keeps only rows whose chunk_id is NOT in chunk_ids.
        """
        import faiss
        import numpy as np

        index, mapping = self._load(user_id)

        if not mapping:
            return

        # Rows to keep
        keep = [i for i, cid in enumerate(mapping) if cid not in chunk_ids]

        if len(keep) == len(mapping):
            return  # nothing was actually removed

        if not keep:
            self._save(user_id, faiss.IndexFlatIP(EMBEDDING_DIM), [])
            return

        # Reconstruct all stored vectors then select only the kept rows.
        # IndexFlatIP stores vectors internally so reconstruct() always works.
        all_vecs = np.vstack(
            [index.reconstruct(i) for i in range(index.ntotal)]
        ).astype(np.float32)
        kept_vecs = all_vecs[keep]
        kept_mapping = [mapping[i] for i in keep]

        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_index.add(kept_vecs)
        self._save(user_id, new_index, kept_mapping)

    def user_has_index(self, user_id: str) -> bool:
        """Return True if the user already has a FAISS index on disk."""
        return self._idx_path(user_id).exists()
