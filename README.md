# 🦅 KFall-Analysis — Fall Detection with CRHNN + FireHawks Optimizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen?logo=pytest)](tests/)

> **Human fall detection and Activity of Daily Living (ADL) classification from
> wearable inertial sensors, powered by a custom Conv-Recurrent HopField Neural
> Network (CRHNN) and a bio-inspired FireHawks gradient-descent optimizer.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Dataset Setup](#-dataset-setup)
- [Running Locally](#-running-locally)
- [Testing](#-testing)
- [Results](#-results)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Falls are a leading cause of injury and death in elderly populations.
This project implements a **wearable-sensor–based fall detection system**
trained on the [KFall dataset](https://www.mdpi.com/1424-8220/21/16/5388),
a rich multi-trial dataset with **9-axis IMU data** (accelerometer, gyroscope,
Euler angles) collected from subjects performing a variety of fall and ADL
scenarios.

### Key innovations

| Component | Description |
|---|---|
| **CRHNN** | Conv1D stem → stacked Bidirectional LSTMs → HopField attention → softmax head |
| **HopField layer** | Modern Hopfield / content-addressable attention over the recurrent state sequence |
| **FireHawks Optimizer** | Bio-inspired RMSProp variant; adaptive per-weight learning rate scaling |
| **Modular codebase** | Fully pip-installable, tested, CLI-driven — ready for CI/CD |

---

## 🏗 Architecture

```
Input  (batch, 1, 9)          ← 9 IMU channels, singleton time-step
   │
Conv1D(128, k=3, relu)        ← short-range temporal feature extraction
   │
BiLSTM(64, return_seq=True)   ┐
BiLSTM(32, return_seq=True)   │  stacked recurrent layers (past + future context)
BiLSTM(16, return_seq=True)   ┘
   │
HopField(units=8)             ← content-addressable attention over all hidden states
   │
Dense(n_classes, softmax)     ← multi-class fall / ADL output
```

### HopField Attention

The HopField layer implements a simplified Modern Hopfield retrieval rule:

1. Project every hidden state through a shared linear layer (keys).
2. Extract the **last** hidden state as the query.
3. Compute dot-product similarity → softmax attention weights.
4. Weighted sum of hidden states → context vector.
5. Concatenate context + query → `tanh` projection → fixed-size embedding.

This produces a **context-enriched, fixed-size representation** regardless of
sequence length, and focuses on the most task-relevant timesteps.

---

## 📁 Project Structure

```
kfall-analysis/
├── src/kfall/                  # installable Python package
│   ├── config.py               # all hyper-parameters & paths (single source of truth)
│   ├── utils.py                # global random seed management
│   ├── data/
│   │   ├── loader.py           # raw KFall data → merged DataFrame
│   │   └── preprocessor.py     # scaling, splitting, one-hot encoding
│   ├── models/
│   │   ├── hopfield.py         # HopField attention Keras layer
│   │   ├── optimizer.py        # FireHawks optimizer (RMSProp variant)
│   │   └── crhnn.py            # CRHNN model builder
│   ├── training/
│   │   ├── trainer.py          # high-level training loop + resume support
│   │   └── callbacks.py        # per-epoch CSV logging + live plot
│   └── evaluation/
│       └── evaluator.py        # metrics, confusion matrix, PR/ROC curves
│
├── scripts/
│   ├── train.py                # end-to-end training pipeline (CLI)
│   ├── evaluate.py             # evaluate a saved checkpoint (CLI)
│   └── predict.py              # batch inference on new sensor CSV (CLI)
│
├── tests/
│   ├── test_data.py            # data loading & preprocessing tests
│   ├── test_model.py           # CRHNN, HopField & callback tests
│   └── test_optimizer.py       # FireHawks optimizer integration tests
│
├── data/
│   ├── raw/                    # ← place KFall dataset here (gitignored)
│   └── README.md               # dataset download instructions
│
├── outputs/                    # auto-created; gitignored
│   ├── models/                 # model.h5, acc_loss.csv, acc_loss.png
│   └── results/
│       ├── Train/              # metrics.csv, conf_mat.png, pr_curve.png, roc_curve.png
│       └── Test/
│
├── setup.py
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1 — Clone

```bash
git clone https://github.com/yourusername/kfall-analysis.git
cd kfall-analysis
```

### 2 — Setup environment and dependencies

We recommend using [uv](https://github.com/astral-sh/uv) for lightning-fast setup. `uv` will automatically download the correct Python version (TensorFlow 2.10 requires Python <= 3.10) and install all dependencies:

```bash
uv sync
```

Alternatively, using standard pip:
```bash
python -m venv .venv
# activate virtual environment, then:
pip install -e .
```

### 4 — Set up the dataset

See [Dataset Setup](#-dataset-setup) below.

### 5 — Train

```bash
uv run kfall-train
```

### 6 — Evaluate

```bash
uv run kfall-evaluate --split Train --split Test
```

### 7 — Predict on new data

```bash
uv run kfall-predict --input my_sensor_data.csv --output predictions.csv
```

---

## 📦 Dataset Setup

1. Download `sensor_data.zip`, `label_data.zip`, and `k_fall.csv` from the
   [KFall dataset page](https://www.mdpi.com/1424-8220/21/16/5388).

2. Place files as follows:

```
data/raw/
├── k_fall.csv
├── sensor_data/
│   └── <SubjectId>/
│       └── ...T...R0....csv
└── label_data/
    └── ...xlsx
```

See [`data/README.md`](data/README.md) for full instructions.

---

## 🚀 Running Locally

### Full training pipeline

```bash
uv run kfall-train \
    --epochs 800 \
    --batch-size 1000 \
    --lr 0.001 \
    --test-size 0.3
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 800 | Total training epochs |
| `--batch-size` | 1000 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--test-size` | 0.3 | Test split fraction |
| `--skip-load` | False | Reuse existing `data/processed/data.csv` |

Training **automatically resumes** from the last checkpoint if `outputs/models/model.h5`
and `outputs/models/acc_loss.csv` already exist.

### Evaluation only

```bash
uv run kfall-evaluate
# or for a single split:
uv run kfall-evaluate --split Test
```

### Batch inference

```bash
uv run kfall-predict \
    --input path/to/sensor.csv \
    --output predictions.csv
```

The input CSV must have these 9 columns (order matters):
`AccX, AccY, AccZ, GyrX, GyrY, GyrZ, EulerX, EulerY, EulerZ`

---

## 🧪 Testing

Run the full test suite (no dataset required — all tests use synthetic data):

```bash
uv run pytest tests/ -v
```

With coverage report:

```bash
uv run pytest tests/ -v --cov=src/kfall --cov-report=term-missing
```

### Test coverage

| Module | Tests |
|---|---|
| `data/preprocessor.py` | output shapes, one-hot encoding, scaler, reproducibility |
| `data/loader.py` | path validation (missing dir / CSV) |
| `models/hopfield.py` | output shape, serialisation, variable seq lengths |
| `models/crhnn.py` | input/output shape, NaN check, layer names, training step |
| `models/optimizer.py` | config serialisation, momentum, loss convergence |
| `training/callbacks.py` | CSV creation, epoch appending, resume |

---

## 📊 Results

After 800 epochs on the full KFall dataset (70 / 30 split):

| Split | Accuracy | Macro F1 |
|---|---|---|
| Train | ~99% | ~0.99 |
| Test | ~95% | ~0.94 |

*Exact figures will vary by run.  Check `outputs/results/` after training.*

Diagnostic plots saved to `outputs/results/<Split>/`:

- `conf_mat.png` — confusion matrix heatmap
- `pr_curve.png` — per-class precision-recall curves
- `roc_curve.png` — per-class ROC curves with AUC scores

---

## ⚙️ Configuration

All hyper-parameters live in a **single file**: [`src/kfall/config.py`](src/kfall/config.py).

```python
from kfall.config import Config

cfg = Config()
cfg.training.epochs = 100        # override any default
cfg.training.batch_size = 512
cfg.model.lstm_units = [64, 32]
```

No magic numbers scattered across files.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and add tests.
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request.

Please follow [PEP 8](https://peps.python.org/pep-0008/) and include
docstrings for all public functions / classes.

---

## 📖 Citation

If you use this code in academic work, please cite:

```bibtex
@dataset{kfall2021,
  author    = {Kim, J. and others},
  title     = {KFall: A Multi-Modal Dataset for Human Fall Detection},
  journal   = {Sensors},
  year      = {2021},
  volume    = {21},
  number    = {16},
  doi       = {10.3390/s21165388}
}
```

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using TensorFlow 2 · Keras · scikit-learn</p>
