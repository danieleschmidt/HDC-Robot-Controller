"""OneShotClassifier: train with one example per class, predict via nearest-HV."""

from __future__ import annotations

import numpy as np

from .vectors import HyperdimensionalVector, bundle
from .memory import ItemMemory
from .encoder import SensorEncoder


class OneShotClassifier:
    """HDC-based classifier that learns from one (or few) examples per class.

    Training:
    - Encode each labelled sensor reading into an HV.
    - Bundle multiple training HVs per class into a prototype HV.
    - Store class prototypes in ItemMemory.

    Prediction:
    - Encode the query sensor reading into an HV.
    - Recall nearest prototype from ItemMemory.

    Args:
        n_sensors:  Number of sensor channels.
        sensor_min: Minimum sensor value (scalar or per-sensor array).
        sensor_max: Maximum sensor value (scalar or per-sensor array).
        n_levels:   Quantisation levels for SensorEncoder.
        dim:        HV dimensionality.
        seed:       Random seed.
    """

    def __init__(
        self,
        n_sensors: int,
        sensor_min: float | np.ndarray = 0.0,
        sensor_max: float | np.ndarray = 1.0,
        n_levels: int = 100,
        dim: int = 10000,
        seed: int = 42,
    ) -> None:
        self.encoder = SensorEncoder(
            n_sensors=n_sensors,
            sensor_min=sensor_min,
            sensor_max=sensor_max,
            n_levels=n_levels,
            dim=dim,
            seed=seed,
        )
        self._memory = ItemMemory()
        # Accumulate training HVs before bundling
        self._class_hvs: dict[str, list[HyperdimensionalVector]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, label: str, readings: np.ndarray | list[float]) -> None:
        """Add one labelled training example.

        Can be called multiple times per class; prototype is updated via
        bundling.
        """
        hv = self.encoder.encode(readings)
        if label not in self._class_hvs:
            self._class_hvs[label] = []
        self._class_hvs[label].append(hv)
        # Recompute prototype as majority-vote bundle of all seen examples
        prototype = bundle(*self._class_hvs[label])
        self._memory.store(label, prototype)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, readings: np.ndarray | list[float]) -> str | None:
        """Predict the class label for a sensor reading."""
        query_hv = self.encoder.encode(readings)
        return self._memory.recall(query_hv)

    def predict_with_score(self, readings: np.ndarray | list[float]) -> tuple[str | None, float]:
        """Predict label and return similarity score."""
        query_hv = self.encoder.encode(readings)
        return self._memory.recall_with_score(query_hv)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        return list(self._class_hvs.keys())

    def __repr__(self) -> str:
        return f"OneShotClassifier(classes={self.classes}, n_sensors={self.encoder.n_sensors})"
