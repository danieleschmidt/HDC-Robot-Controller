#!/usr/bin/env python3
"""Demo: HDC gesture classification on synthetic sensor data.

Scenario:
    5 gesture classes, each with a distinctive pattern across 6 sensors.
    Train with 3 examples per class (with small noise), then test on 15+ new samples.
    Expect accuracy > 80%.
"""

import numpy as np
from hdc import OneShotClassifier

# Reproducible RNG
rng = np.random.default_rng(0)

N_SENSORS = 6
GESTURES = {
    "swipe_left":  np.array([0.1, 0.9, 0.1, 0.1, 0.5, 0.2]),
    "swipe_right": np.array([0.9, 0.1, 0.1, 0.1, 0.5, 0.2]),
    "swipe_up":    np.array([0.5, 0.5, 0.9, 0.1, 0.2, 0.3]),
    "swipe_down":  np.array([0.5, 0.5, 0.1, 0.9, 0.2, 0.3]),
    "tap":         np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.8]),
}

NOISE_TRAIN = 0.05
NOISE_TEST  = 0.08
TRAIN_PER_CLASS = 3
TEST_PER_CLASS  = 4  # 5 classes × 4 = 20 test samples


def add_noise(base: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(base + rng.normal(0, scale, size=base.shape), 0.0, 1.0)


def main() -> None:
    clf = OneShotClassifier(
        n_sensors=N_SENSORS,
        sensor_min=0.0,
        sensor_max=1.0,
        n_levels=100,
        dim=10000,
        seed=42,
    )

    # ── Training ──────────────────────────────────────────────────────
    print("Training...")
    for label, base in GESTURES.items():
        for _ in range(TRAIN_PER_CLASS):
            clf.train(label, add_noise(base, NOISE_TRAIN, rng))
    print(f"  Classes: {clf.classes}\n")

    # ── Testing ───────────────────────────────────────────────────────
    correct = 0
    total = 0
    print(f"{'True':>12}  {'Predicted':>12}  {'Score':>6}  {'OK'}")
    print("-" * 46)
    for label, base in GESTURES.items():
        for _ in range(TEST_PER_CLASS):
            reading = add_noise(base, NOISE_TEST, rng)
            pred, score = clf.predict_with_score(reading)
            ok = "✓" if pred == label else "✗"
            print(f"{label:>12}  {str(pred):>12}  {score:.4f}  {ok}")
            correct += int(pred == label)
            total += 1

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
    assert accuracy > 0.80, f"Accuracy {accuracy:.1%} below 80% threshold!"
    print("Demo passed ✓")


if __name__ == "__main__":
    main()
