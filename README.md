# MIT-BIH Federated Learning Repository

A Python library to explore, analyze, and train deep learning models on MIT-BIH datasets.

<p align="left">
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.14+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/Astral_uv-0.11+-de5d68?style=for-the-badge&logo=astral&logoColor=white" alt="Astral uv"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit_learn-1.9+-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://pandas.pydata.org"><img src="https://img.shields.io/badge/Pandas-3.0+-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-2.5+-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/Ruff-Formatter-black?style=for-the-badge&logo=ruff&logoColor=white" alt="Ruff"></a>
  <a href="https://code.visualstudio.com"><img src="https://img.shields.io/badge/VS_Code-IDE-0078d4?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VS Code"></a>
</p>

## Prerequisites

Ensure you have the following installed on your system:
- **Git** (for cloning the repository)
- [**uv**](https://github.com/astral-sh/uv) (for ultra-fast Python package and environment management)
- [**just**](https://github.com/casey/just) (optional, command runner for workspace workflows)

---

## Getting Started

### 1. Clone the Repository

Clone the project to your local machine and navigate into the directory:

```bash
git clone https://github.com/bernardbdas/mit-bih.git
cd mit-bih
```

### 2. Environment Setup & Dependency Synchronization

You can synchronize dependencies and set up the default virtual environment (`.venv`) using **`uv`** directly or via the provided **`justfile`** targets:

#### Option A: Using `just` (Recommended)
```bash
# 1. Initialize the virtual environment (.venv) using Python 3.14
just init

# 2. Sync all project dependencies
just sync
```

#### Option B: Using `uv` directly
```bash
# Sync dependencies and set up the default virtual environment
uv sync
```

#### Other Workspace Tasks (`just`)
* **Lock dependencies**: `just lock` (runs `uv lock`)
* **Wipe virtual environment**: `just clean` (deletes `.venv/`)
---

## Dataset Download & Management

This library uses the **`wfdb`** Python library to download, read, and write electrocardiogram (ECG) data from PhysioNet. We provide an automated helper script [scripts/download_datasets.py](file:///Users/bernard/Developer/FORKS/mit-bih/scripts/download_datasets.py) to download any of the supported MIT-BIH and other ECG datasets into [data/raw/](file:///Users/bernard/Developer/FORKS/mit-bih/data/raw/).

### 1. List Available Datasets
To view all supported datasets and their corresponding PhysioNet slugs:
```bash
uv run scripts/download_datasets.py --list
```

### 2. Supported Datasets & Slugs

| Dataset Name | PhysioNet Slug |
| :--- | :--- |
| **MIT-BIH Arrhythmia Database** | `mitdb` |
| **MIT-BIH Arrhythmia DB P-wave Annotations** | `pwave` |
| **MIT-BIH Atrial Fibrillation Database** | `afdb` |
| **MIT-BIH Long-term ECG Database** | `ltdb` |
| **MIT-BIH Supraventricular Arrhythmia Database** | `svdb` |
| **MIT-BIH ST Change Database** | `stdb` |
| **MIT-BIH ECG Compression Database** | `cdb` |
| **MIT-BIH Malignant Ventricular Ectopy Database** | `vfdb` |
| **MIT-BIH Noise Stress Test Database** | `nstdb` |
| **MIT-BIH Normal Sinus Rhythm Database** | `nsrdb` |
| **Recordings excluded from MIT-BIH NSR DB** | `nsr2db` |
| **Sudden Cardiac Death Holter Database** | `sddb` |
| **Abdominal and Direct Fetal ECG Database** | `adfecgdb` |
| **Non-Invasive Fetal ECG Arrhythmia Database** | `nifecgdb` |
| **MIT-BIH Polysomnographic Database** | `slpdb` |
| **ECG Fragment Database** | `ecg-fragment-high-risk-label` |
| **European ST-T Database** | `edb` |

### 3. Run Download Command

* **To download a specific dataset** (e.g. `mitdb`):
  ```bash
  uv run scripts/download_datasets.py --db mitdb
  ```
  This will save all files (`.hea`, `.dat`, `.atr`, etc.) in `data/raw/mitdb/`.

* **To download all supported datasets** sequentially:
  ```bash
  uv run scripts/download_datasets.py --db all
  ```
  > [!WARNING]
  > Downloading all datasets will require significant disk space (several gigabytes) and internet bandwidth.

---

## Development & Verification

### Running Tests

You can run the unit test suite with:

```bash
# Using standard Python/uv
uv run python -m unittest discover -s tests
```

### VS Code Integration

This repository includes custom VS Code configurations (`.vscode/` directory) to enhance your workflow:
- **Recommended Extensions**: Automations for Ruff (formatting and linting on save), Pylance, Jupyter, and Justfile syntax highlighting.
- **Run Tasks**: Run any of the `just` commands or the test suite directly from the Command Palette (`Cmd+Shift+P` -> `Tasks: Run Task`).
- **Debugging**: Launch configurations for debugging the current file, unit tests, or the `mit_bih` module (`python -m mit_bih`).
