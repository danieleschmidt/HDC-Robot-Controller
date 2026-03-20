# HDC Robot Controller

A pure-Python implementation of **Hyperdimensional Computing (HDC)** for robot sensor classification. No deep learning, no GPU — just high-dimensional binary algebra.

## What is HDC?

HDC encodes information as random binary vectors of 10 000 dimensions. Patterns are learned by XOR-binding and majority-vote bundling — operations that are fast, noise-tolerant, and one-shot capable.

## Installation

```bash
pip install numpy
pip install -e .
```

## Quick Start

```python
from hdc import OneShotClassifier
import numpy as np

clf = OneShotClassifier(n_sensors=6, sensor_min=0.0, sensor_max=1.0)
clf.train("swipe_left",  [0.1, 0.9, 0.1, 0.1, 0.5, 0.2])
clf.train("swipe_right", [0.9, 0.1, 0.1, 0.1, 0.5, 0.2])

print(clf.predict([0.12, 0.88, 0.09, 0.11, 0.51, 0.19]))  # → swipe_left
```

## Package Layout

```
hdc/
  vectors.py     — HyperdimensionalVector (XOR bind, majority bundle, cosine sim)
  memory.py      — ItemMemory (store/recall by similarity)
  encoder.py     — SensorEncoder (level-hypervectors → HV)
  classifier.py  — OneShotClassifier (train/predict)
demo.py          — Gesture classification demo (5 classes × 6 sensors)
tests/           — 15+ pytest tests
```

## Core Concepts

| Concept | Operation | Property |
|---------|-----------|----------|
| **Binding** | XOR | Invertible: `bind(bind(a,b), b) == a` |
| **Bundling** | Majority vote | Result similar to all inputs |
| **Similarity** | Cosine / Hamming | Random vectors ≈ 0.5, identical = 1.0 |
| **Encoding** | Level-HVs + position-HVs | Similar inputs → similar HVs |

## Run the Demo

```bash
python demo.py
```

Expected output: 5-class gesture classifier, accuracy > 80%.

## Run Tests

```bash
pytest tests/
```

## License

MIT
