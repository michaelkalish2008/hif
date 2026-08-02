"""Embedding generation for text outputs using EmbeddingGemma or a fallback sentence-transformer.

The EmbeddingModel wraps sentence-transformers ONLY for converting token strings to vectors
for clustering. It is NOT the model under analysis (GPT-2 / HFModel).
"""

import hashlib
import logging
from pathlib import Path

import numpy as np

from hif.config import EmbeddingConfig
from hif.utils.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_DIM = 384   # all-MiniLM-L6-v2 native dim


class EmbeddingModel:
    """Wraps a sentence-transformer for embedding token strings.

    Lazy-loads the backend; falls back to MiniLM if the primary model fails.
    Caches results to disk by content hash.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None          # loaded lazily; tests may inject directly
        self._model_name: str = ""
        self._embedding_dim: int = 0
        self._cache_dir: Path = Path(config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the primary model; fall back to MiniLM on any failure."""
        # Lazy import so this module is importable without sentence_transformers.
        try:
            import sentence_transformers  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for EmbeddingModel. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        primary = self._config.model_name
        try:
            if self._config.matryoshka_dim is not None:
                # Pass truncate_dim for Matryoshka if supported (sentence-transformers >= 3.x).
                try:
                    model = SentenceTransformer(
                        primary, truncate_dim=self._config.matryoshka_dim
                    )
                    dim = self._config.matryoshka_dim
                except TypeError:
                    # Older sentence-transformers without truncate_dim support.
                    model = SentenceTransformer(primary)
                    dim = self._config.matryoshka_dim  # will slice manually in encode
            else:
                model = SentenceTransformer(primary)
                # renamed in newer sentence-transformers; support both
                get_dim = getattr(model, "get_embedding_dimension", None) or (
                    model.get_sentence_embedding_dimension
                )
                dim = int(get_dim())

            self._model = model
            self._model_name = primary
            self._embedding_dim = dim
            logger.debug("Loaded embedding model: %s (dim=%d)", primary, dim)
        except Exception as exc:  # noqa: BLE001
            fallback = self._config.fallback_model_name
            # A fallback on defaults should never happen (primary == fallback);
            # an unexpected fallback (primary != fallback) changes what the
            # similarity/exposure numbers mean, so it stays visible by default.
            level = logging.WARNING if primary != fallback else logging.DEBUG
            logger.log(
                level,
                "Failed to load primary embedding model %r (%s). "
                "Falling back to %r.",
                primary,
                exc,
                fallback,
            )
            self._model = SentenceTransformer(fallback)
            self._model_name = fallback
            self._embedding_dim = _FALLBACK_DIM
            logger.debug("Loaded fallback embedding model: %s (dim=%d)", fallback, _FALLBACK_DIM)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        raw = f"{self._model_name}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                vec = np.load(str(p))
                # Discard cache entries whose dimension doesn't match the loaded
                # model.  This guards against stale entries written by a
                # different model (e.g. primary vs. fallback swap).
                if vec.shape != (self._embedding_dim,):
                    logger.debug(
                        "Discarding cached embedding with wrong shape %s "
                        "(expected (%d,)); will re-encode.",
                        vec.shape,
                        self._embedding_dim,
                    )
                    return None
                return vec
            except Exception:  # noqa: BLE001
                return None
        return None

    def _save_cached(self, key: str, vec: np.ndarray) -> None:
        try:
            np.save(str(self._cache_path(key)), vec)
        except Exception:  # noqa: BLE001
            pass  # cache write failures are non-fatal

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Actual model name that was loaded (primary or fallback)."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension after any Matryoshka truncation."""
        return self._embedding_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns shape (N, embedding_dim).

        Uses disk cache; only calls the model for uncached texts.
        """
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        # Check cache for each text.
        results: list[np.ndarray | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        keys = [self._cache_key(t) for t in texts]

        for i, (text, key) in enumerate(zip(texts, keys)):
            cached = self._load_cached(key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch-encode uncached texts.
        if uncached_texts:
            raw = self._model.encode(uncached_texts, show_progress_bar=False)
            raw = np.array(raw, dtype=np.float32)
            # Slice to matryoshka_dim if the model didn't handle it internally.
            if raw.shape[1] > self._embedding_dim:
                raw = raw[:, : self._embedding_dim]

            # Map results back by original index.
            for offset, idx in enumerate(uncached_indices):
                vec = raw[offset]
                results[idx] = vec
                self._save_cached(keys[idx], vec)

        output = np.stack([r for r in results], axis=0)
        return output.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (embedding_dim,)."""
        return self.embed([text])[0]
