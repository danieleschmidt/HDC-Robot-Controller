"""15+ pytest tests for the HDC library."""

import numpy as np
import pytest

from hdc.vectors import HyperdimensionalVector, bundle
from hdc.memory import ItemMemory
from hdc.encoder import SensorEncoder
from hdc.classifier import OneShotClassifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def hv_a(rng):
    return HyperdimensionalVector(rng=rng, dim=10000)


@pytest.fixture
def hv_b(rng):
    return HyperdimensionalVector(rng=rng, dim=10000)


@pytest.fixture
def hv_c(rng):
    return HyperdimensionalVector(rng=rng, dim=10000)


# ── 1. HyperdimensionalVector creation & shape ─────────────────────────────

def test_hv_creation_shape():
    hv = HyperdimensionalVector(dim=10000)
    assert hv.data.shape == (10000,)


def test_hv_binary_values():
    hv = HyperdimensionalVector(dim=10000)
    assert set(np.unique(hv.data)).issubset({0, 1})


def test_hv_default_dim():
    hv = HyperdimensionalVector()
    assert hv.dim == 10000


def test_hv_custom_data():
    data = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    hv = HyperdimensionalVector(data=data)
    assert np.array_equal(hv.data, data)


# ── 2. XOR binding is its own inverse ─────────────────────────────────────

def test_bind_inverse(hv_a, hv_b):
    """bind(bind(a, b), b) == a  (exactly, for binary XOR)."""
    ab = hv_a.bind(hv_b)
    recovered = ab.bind(hv_b)
    assert hv_a == recovered


def test_bind_commutativity(hv_a, hv_b):
    """XOR is commutative: a XOR b == b XOR a."""
    assert hv_a.bind(hv_b) == hv_b.bind(hv_a)


def test_bind_associativity(hv_a, hv_b, hv_c):
    """XOR associativity: (a XOR b) XOR c == a XOR (b XOR c)."""
    lhs = hv_a.bind(hv_b).bind(hv_c)
    rhs = hv_a.bind(hv_b.bind(hv_c))
    assert lhs == rhs


def test_bind_self_is_zero(hv_a):
    """a XOR a = all-zeros vector."""
    result = hv_a.bind(hv_a)
    assert np.all(result.data == 0)


# ── 3. Similarity ─────────────────────────────────────────────────────────

def test_similarity_identical(hv_a):
    """Cosine similarity of a vector with itself is 1.0."""
    assert hv_a.similarity(hv_a) == pytest.approx(1.0, abs=1e-6)


def test_similarity_random_approx_half(hv_a, hv_b):
    """Two random independent HVs should have similarity ≈ 0.5."""
    sim = hv_a.similarity(hv_b)
    assert 0.45 <= sim <= 0.55, f"Expected similarity ≈ 0.5, got {sim:.4f}"


def test_similarity_range(hv_a, hv_b):
    """Similarity is always in [0, 1]."""
    sim = hv_a.similarity(hv_b)
    assert 0.0 <= sim <= 1.0


# ── 4. Bundling ───────────────────────────────────────────────────────────

def test_bundle_similar_to_inputs():
    """Bundle of 3 similar vectors is more similar to each than to a random one."""
    rng = np.random.default_rng(0)
    base = HyperdimensionalVector(rng=rng, dim=10000)
    # Create slightly perturbed copies
    def perturb(hv, n_flips=500, rng=rng):
        d = hv.data.copy()
        idx = rng.choice(hv.dim, size=n_flips, replace=False)
        d[idx] ^= 1
        return HyperdimensionalVector(data=d)

    v1 = perturb(base)
    v2 = perturb(base)
    v3 = perturb(base)
    result = bundle(v1, v2, v3)
    random_hv = HyperdimensionalVector(rng=rng, dim=10000)

    sim_to_v1 = result.similarity(v1)
    sim_to_random = result.similarity(random_hv)
    assert sim_to_v1 > sim_to_random, (
        f"Bundle should be closer to its inputs ({sim_to_v1:.4f}) "
        f"than to a random vector ({sim_to_random:.4f})"
    )


def test_bundle_commutativity(hv_a, hv_b, hv_c):
    """Bundling is order-independent (majority vote)."""
    b1 = bundle(hv_a, hv_b, hv_c)
    b2 = bundle(hv_c, hv_a, hv_b)
    assert b1 == b2


def test_bundle_single():
    """Bundle of one HV equals itself."""
    hv = HyperdimensionalVector(dim=10000)
    result = bundle(hv)
    assert result == hv


# ── 5. ItemMemory ─────────────────────────────────────────────────────────

def test_itemmemory_store_recall():
    """Exact HV recall from single-item memory."""
    mem = ItemMemory()
    hv = HyperdimensionalVector(dim=10000)
    mem.store("cat", hv)
    assert mem.recall(hv) == "cat"


def test_itemmemory_multiple_items():
    """Recall returns the correct argmax when multiple items stored."""
    rng = np.random.default_rng(7)
    mem = ItemMemory()
    hvs = {label: HyperdimensionalVector(rng=rng, dim=10000) for label in ["apple", "banana", "cherry"]}
    for label, hv in hvs.items():
        mem.store(label, hv)

    for label, hv in hvs.items():
        assert mem.recall(hv) == label


def test_itemmemory_empty_returns_none():
    mem = ItemMemory()
    query = HyperdimensionalVector(dim=10000)
    assert mem.recall(query) is None


# ── 6. SensorEncoder ─────────────────────────────────────────────────────

def test_sensorencoder_output_dim():
    enc = SensorEncoder(n_sensors=6, dim=10000, seed=1)
    result = enc.encode([0.1, 0.5, 0.3, 0.7, 0.9, 0.2])
    assert result.dim == 10000


def test_sensorencoder_binary_output():
    enc = SensorEncoder(n_sensors=6, dim=10000, seed=1)
    result = enc.encode([0.0, 0.25, 0.5, 0.75, 1.0, 0.33])
    assert set(np.unique(result.data)).issubset({0, 1})


def test_sensorencoder_deterministic():
    """Same input → same output (deterministic by seed)."""
    enc = SensorEncoder(n_sensors=4, dim=10000, seed=42)
    readings = [0.1, 0.4, 0.7, 0.9]
    hv1 = enc.encode(readings)
    hv2 = enc.encode(readings)
    assert hv1 == hv2


def test_sensorencoder_different_inputs_different_hvs():
    """Different sensor readings should produce different HVs."""
    enc = SensorEncoder(n_sensors=4, dim=10000, seed=42)
    hv1 = enc.encode([0.1, 0.2, 0.3, 0.4])
    hv2 = enc.encode([0.9, 0.8, 0.7, 0.6])
    assert hv1 != hv2


# ── 7. OneShotClassifier ──────────────────────────────────────────────────

def test_classifier_train_predict_single():
    """Train on one class, predict it back."""
    clf = OneShotClassifier(n_sensors=3, sensor_min=0.0, sensor_max=1.0, seed=0)
    clf.train("gesture_A", [0.1, 0.5, 0.9])
    assert clf.predict([0.1, 0.5, 0.9]) == "gesture_A"


def test_classifier_multiple_classes():
    """Classifier correctly distinguishes between 3 well-separated classes."""
    rng = np.random.default_rng(99)
    clf = OneShotClassifier(n_sensors=4, sensor_min=0.0, sensor_max=1.0, seed=5)

    classes = {
        "A": np.array([0.1, 0.1, 0.9, 0.9]),
        "B": np.array([0.9, 0.1, 0.1, 0.9]),
        "C": np.array([0.5, 0.9, 0.5, 0.1]),
    }

    for label, base in classes.items():
        for _ in range(5):
            noisy = np.clip(base + rng.normal(0, 0.03, size=4), 0, 1)
            clf.train(label, noisy)

    correct = 0
    n_test = 10
    for label, base in classes.items():
        for _ in range(n_test):
            noisy = np.clip(base + rng.normal(0, 0.05, size=4), 0, 1)
            if clf.predict(noisy) == label:
                correct += 1

    accuracy = correct / (len(classes) * n_test)
    assert accuracy >= 0.80, f"Expected ≥80% accuracy, got {accuracy:.1%}"


# ── 8. End-to-end: gesture demo ───────────────────────────────────────────

def test_gesture_demo_accuracy():
    """Full gesture-demo scenario: 5 classes × 6 sensors, accuracy > 80%."""
    rng_data = np.random.default_rng(0)

    GESTURES = {
        "swipe_left":  np.array([0.1, 0.9, 0.1, 0.1, 0.5, 0.2]),
        "swipe_right": np.array([0.9, 0.1, 0.1, 0.1, 0.5, 0.2]),
        "swipe_up":    np.array([0.5, 0.5, 0.9, 0.1, 0.2, 0.3]),
        "swipe_down":  np.array([0.5, 0.5, 0.1, 0.9, 0.2, 0.3]),
        "tap":         np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.8]),
    }

    clf = OneShotClassifier(
        n_sensors=6, sensor_min=0.0, sensor_max=1.0,
        n_levels=100, dim=10000, seed=42,
    )

    for label, base in GESTURES.items():
        for _ in range(3):
            noisy = np.clip(base + rng_data.normal(0, 0.05, size=6), 0, 1)
            clf.train(label, noisy)

    correct = 0
    total = 0
    for label, base in GESTURES.items():
        for _ in range(4):
            noisy = np.clip(base + rng_data.normal(0, 0.08, size=6), 0, 1)
            if clf.predict(noisy) == label:
                correct += 1
            total += 1

    accuracy = correct / total
    assert accuracy > 0.80, f"Demo accuracy {accuracy:.1%} < 80%"
