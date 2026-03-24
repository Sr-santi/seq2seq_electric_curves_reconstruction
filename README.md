# Seq2Seq Electric Curves Reconstruction

This project reconstructs customer electricity consumption curves using sequence-to-sequence (Seq2Seq) PyTorch models. The core analysis and training are contained within a Jupyter Notebook.

## Project Structure

- `customer_electricity_consumption_challenge.ipynb`: The main notebook for data exploration, preprocessing, and model training.
- `pyproject.toml` & `uv.lock`: Configuration and exact locked versions of project dependencies.
- `.venv/`: The isolated virtual environment (generated automatically).

## Local Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environments.

### 1. Install `uv`

If you don't already have `uv` installed, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Data Setup

The notebook relies on local CSV files for training and testing that are inside this project

You can use the interface that VS Code or Cursor offers to open and interact with the jupyter notebook or open it in your browser

```bash
uv run jupyter lab
```
