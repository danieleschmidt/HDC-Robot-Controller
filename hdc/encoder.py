"""SensorEncoder: map float sensor arrays to hyperdimensional vectors.

Strategy: level-hypervectors
- The sensor range [min, max] is divided into L evenly-spaced levels.
- Each level k has a pre-generated random HV (level_hvs[k]).
- Adjacent level HVs are constructed by flipping bits progressively so that
  similar sensor values produce similar (high-overlap) HVs.
- For a reading x, find the closest level → return that level's HV.
- For a multi-sensor array, XOR-bind each sensor's HV with a position HV,
  then bundle all sensor HVs into a single output HV.
"""

from __future__ import annotations

import numpy as np

from .vectors import HyperdimensionalVector, bundle, DEFAULT_DIM


class SensorEncoder:
    """Encode a fixed-length float sensor array into an HV.

    Args:
        n_sensors:   Number of sensor channels.
        sensor_min:  Minimum expected sensor value (scalar or array per sensor).
        sensor_max:  Maximum expected sensor value (scalar or array per sensor).
        n_levels:    Number of quantisation levels per sensor.
        dim:         HV dimensionality (default 10000).
        seed:        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_sensors: int,
        sensor_min: float | np.ndarray = 0.0,
        sensor_max: float | np.ndarray = 1.0,
        n_levels: int = 100,
        dim: int = DEFAULT_DIM,
        seed: int = 42,
    ) -> None:
        self.n_sensors = n_sensors
        self.dim = dim
        self.n_levels = n_levels

        rng = np.random.default_rng(seed)

        # Per-sensor min/max (broadcast scalars to arrays)
        self.sensor_min = np.broadcast_to(np.asarray(sensor_min, dtype=np.float64), (n_sensors,)).copy()
        self.sensor_max = np.broadcast_to(np.asarray(sensor_max, dtype=np.float64), (n_sensors,)).copy()

        # Position HVs — one per sensor channel (random, independent)
        self._position_hvs: list[HyperdimensionalVector] = [
            HyperdimensionalVector(rng=rng, dim=dim) for _ in range(n_sensors)
        ]

        # Level HVs — one set per sensor, built via progressive bit-flipping
        # so consecutive levels share high similarity.
        self._level_hvs: list[list[HyperdimensionalVector]] = []
        for _ in range(n_sensors):
            levels = _build_level_hvs(n_levels, dim, rng)
            self._level_hvs.append(levels)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, readings: np.ndarray | list[float]) -> HyperdimensionalVector:
        """Encode a sensor array into a single HV.

        Args:
            readings: Array-like of length n_sensors.

        Returns:
            A single HyperdimensionalVector representing the sensor state.
        """
        readings = np.asarray(readings, dtype=np.float64)
        if readings.shape != (self.n_sensors,):
            raise ValueError(f"Expected {self.n_sensors} sensor values, got {readings.shape}")

        bound_hvs: list[HyperdimensionalVector] = []
        for i, val in enumerate(readings):
            level_idx = self._value_to_level(val, i)
            level_hv = self._level_hvs[i][level_idx]
            # Bind with position HV to distinguish sensor channels
            bound = level_hv.bind(self._position_hvs[i])
            bound_hvs.append(bound)

        # Bundle all bound HVs into one
        return bundle(*bound_hvs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _value_to_level(self, value: float, sensor_idx: int) -> int:
        lo = self.sensor_min[sensor_idx]
        hi = self.sensor_max[sensor_idx]
        if hi == lo:
            return 0
        # Clip and map to [0, n_levels-1]
        clipped = float(np.clip(value, lo, hi))
        level = int((clipped - lo) / (hi - lo) * (self.n_levels - 1))
        return min(level, self.n_levels - 1)


def _build_level_hvs(n_levels: int, dim: int, rng: np.random.Generator) -> list[HyperdimensionalVector]:
    """Build n_levels HVs where adjacent levels differ by ~1/n_levels of bits.

    Level 0 is random; each subsequent level flips dim//n_levels bits from the previous.
    This creates a smooth gradient of similarity across levels.
    """
    # Start with a random base
    base = rng.integers(0, 2, size=dim, dtype=np.uint8)
    # Choose which bit positions to flip at each step (pre-generate)
    # We need (n_levels - 1) * flip_count flips total; use a permutation
    flip_count = max(1, dim // n_levels)
    all_positions = rng.permutation(dim)

    levels: list[HyperdimensionalVector] = []
    current = base.copy()
    levels.append(HyperdimensionalVector(data=current.copy()))

    for step in range(1, n_levels):
        start = (step - 1) * flip_count
        end = start + flip_count
        # Wrap around if we run out of positions
        positions = all_positions[start % dim: end % dim if end % dim != 0 else dim]
        if len(positions) == 0:
            positions = all_positions[:flip_count]
        current[positions] ^= 1  # flip those bits
        levels.append(HyperdimensionalVector(data=current.copy()))

    return levels
