"""Hyperdimensional vectors: binary {0,1}^N with HDC operations."""

from __future__ import annotations

import numpy as np

DEFAULT_DIM = 10000


class HyperdimensionalVector:
    """A binary hyperdimensional vector of dimension N.

    Supports:
    - XOR binding (invertible association)
    - Majority-vote bundling (set union)
    - Cosine similarity (distance measure)
    - Hamming similarity (1 - hamming_distance / N)
    """

    def __init__(self, data: np.ndarray | None = None, dim: int = DEFAULT_DIM, rng: np.random.Generator | None = None) -> None:
        """Create an HV.

        Args:
            data:  Pre-built binary array of shape (dim,). If None, a random HV is generated.
            dim:   Dimensionality (used only when data is None).
            rng:   Optional numpy random generator for reproducibility.
        """
        if data is not None:
            self._v = np.asarray(data, dtype=np.uint8)
        else:
            _rng = rng if rng is not None else np.random.default_rng()
            self._v = _rng.integers(0, 2, size=dim, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self._v)

    @property
    def data(self) -> np.ndarray:
        return self._v

    # ------------------------------------------------------------------
    # HDC operations
    # ------------------------------------------------------------------

    def bind(self, other: "HyperdimensionalVector") -> "HyperdimensionalVector":
        """XOR binding — invertible association of two HVs.

        bind(bind(a, b), b) == a  (exactly, for binary vectors)
        """
        return HyperdimensionalVector(data=np.bitwise_xor(self._v, other._v))

    def similarity(self, other: "HyperdimensionalVector") -> float:
        """Cosine similarity (equivalent to 1 - normalised Hamming for binary {0,1} vectors).

        Returns a value in [0, 1] where 1 means identical.
        """
        # Convert to {-1, +1} for cosine calculation
        a = self._v.astype(np.float32) * 2 - 1
        b = other._v.astype(np.float32) * 2 - 1
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # Map from [-1,1] to [0,1]
        return (dot / (norm_a * norm_b) + 1.0) / 2.0

    def hamming_similarity(self, other: "HyperdimensionalVector") -> float:
        """1 - (Hamming distance / dim).  Identical vectors → 1.0."""
        matches = int(np.count_nonzero(self._v == other._v))
        return matches / self.dim

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __xor__(self, other: "HyperdimensionalVector") -> "HyperdimensionalVector":
        return self.bind(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperdimensionalVector):
            return NotImplemented
        return bool(np.array_equal(self._v, other._v))

    def __repr__(self) -> str:
        return f"HyperdimensionalVector(dim={self.dim}, ones={int(self._v.sum())})"


# ------------------------------------------------------------------
# Bundling (majority vote)
# ------------------------------------------------------------------


def bundle(*hvs: HyperdimensionalVector) -> HyperdimensionalVector:
    """Majority-vote bundling of multiple HVs.

    For each bit position, the result is 1 if more vectors have 1 than 0;
    0 if more have 0 than 1; tie → 0.

    The bundled vector is similar to each input HV.
    """
    if not hvs:
        raise ValueError("bundle() requires at least one HyperdimensionalVector")
    dim = hvs[0].dim
    stack = np.stack([hv.data for hv in hvs], axis=0).astype(np.int32)  # shape (K, dim)
    counts = stack.sum(axis=0)  # number of 1s per position
    threshold = len(hvs) / 2.0
    result = (counts > threshold).astype(np.uint8)
    return HyperdimensionalVector(data=result)
