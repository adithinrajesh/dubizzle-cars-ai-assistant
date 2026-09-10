"""Atomic JSON cache, keyed by stable listing ID and canonical document digest."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .documents import DOCUMENT_VERSION
from .embeddings import EmbeddingProvider, Vector, embed_documents, normalize_vector
from .errors import EmbeddingCacheError, EmbeddingError

CACHE_VERSION = 1


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """A single-writer cache. Corruption raises; version/identity changes regenerate."""

    def __init__(self, path: str | Path = ".cache/inventory_embeddings.json") -> None:
        self.path = Path(path)

    def _read(self, provider_id: str, dimension: int) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise EmbeddingCacheError("Cannot read embedding cache; remove it to rebuild") from exc
        try:
            if not isinstance(payload, dict):
                raise ValueError("expected object")
            if (
                payload["cache_version"] != CACHE_VERSION
                or payload["document_version"] != DOCUMENT_VERSION
                or payload["provider_id"] != provider_id
            ):
                return {}
            if payload["dimension"] != dimension:
                raise ValueError("cached dimension disagrees with query embedding")
            entries = payload["entries"]
            if not isinstance(entries, dict) or payload["record_count"] != len(entries):
                raise ValueError("record count mismatch")
            if payload["checksum"] != _digest(entries):
                raise ValueError("cache checksum mismatch")
            hashes = {key: entry["document_hash"] for key, entry in entries.items()}
            if payload["inventory_hash"] != _digest(hashes):
                raise ValueError("inventory hash mismatch")
            for listing_id, entry in entries.items():
                if not listing_id or not isinstance(entry["document_hash"], str):
                    raise ValueError("invalid listing ID or document hash")
                normalize_vector(entry["vector"], dimension)
            return entries
        except (KeyError, TypeError, ValueError, EmbeddingError) as exc:
            raise EmbeddingCacheError("Inconsistent embedding cache; remove it to rebuild") from exc

    def _write(self, entries: dict[str, Any], provider_id: str, dimension: int) -> None:
        payload = {
            "cache_version": CACHE_VERSION,
            "document_version": DOCUMENT_VERSION,
            "provider_id": provider_id,
            "dimension": dimension,
            "record_count": len(entries),
            "inventory_hash": _digest(
                {key: entry["document_hash"] for key, entry in entries.items()}
            ),
            "checksum": _digest(entries),
            "entries": entries,
        }
        temporary: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(payload, file, sort_keys=True, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, self.path)
        except (OSError, ValueError) as exc:
            raise EmbeddingCacheError("Cannot write embedding cache") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get_embeddings(
        self,
        documents: Mapping[str, str],
        provider: EmbeddingProvider,
        provider_id: str,
        dimension: int,
    ) -> dict[str, Vector]:
        """Only embed missing/changed documents; cached noncandidates are never returned."""
        entries = self._read(provider_id, dimension)
        hashes = {key: _digest(text) for key, text in documents.items()}
        missing = sorted(
            key
            for key in documents
            if key not in entries or entries[key]["document_hash"] != hashes[key]
        )
        if missing:
            vectors = embed_documents(provider, [documents[key] for key in missing], dimension)
            for key, vector in zip(missing, vectors, strict=True):
                entries[key] = {"document_hash": hashes[key], "vector": vector}
            self._write(entries, provider_id, dimension)
        return {key: tuple(entries[key]["vector"]) for key in documents}
