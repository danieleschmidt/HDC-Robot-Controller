"""ItemMemory: associative memory for HVs."""

from __future__ import annotations

from typing import Any

from .vectors import HyperdimensionalVector


class ItemMemory:
    """Associative memory that maps string labels to HyperdimensionalVectors.

    Usage::

        mem = ItemMemory()
        mem.store("cat", cat_hv)
        mem.store("dog", dog_hv)
        label = mem.recall(query_hv)  # → "cat" or "dog"
    """

    def __init__(self) -> None:
        self._items: dict[str, HyperdimensionalVector] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def store(self, label: str, vector: HyperdimensionalVector) -> None:
        """Store (or overwrite) a label→vector mapping."""
        self._items[label] = vector

    def recall(self, query: HyperdimensionalVector) -> str | None:
        """Return the label whose stored HV is most similar to query.

        Returns None if the memory is empty.
        """
        if not self._items:
            return None
        best_label: str | None = None
        best_score = -1.0
        for label, hv in self._items.items():
            score = query.similarity(hv)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def recall_with_score(self, query: HyperdimensionalVector) -> tuple[str | None, float]:
        """Return (label, similarity_score) for the closest stored HV."""
        if not self._items:
            return None, 0.0
        best_label: str | None = None
        best_score = -1.0
        for label, hv in self._items.items():
            score = query.similarity(hv)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, label: Any) -> bool:
        return label in self._items

    def __repr__(self) -> str:
        labels = list(self._items.keys())
        return f"ItemMemory(labels={labels})"
