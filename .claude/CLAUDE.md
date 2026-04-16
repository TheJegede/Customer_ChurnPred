# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A data science project for analyzing, visualizing, and modeling data. It includes a Python package (`src/`), Jupyter notebooks, a serving layer (FastAPI or Streamlit via `app/`), and experiment tracking via MLflow.

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # installs the src/ package in editable mode
```

## Common Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_something.py

# Run a single test by name
pytest tests/test_something.py::test_function_name

# Start Streamlit app
streamlit run app/app.py

# Start FastAPI server
uvicorn app.main:app --reload

# Start MLflow UI
mlflow ui
```

## Architecture

The project uses a `src/` layout — the installable package is `ml_project`, installed via `pip install -e .`.

- `src/data/` — data loading, ingestion, and preprocessing logic
- `src/features/` — feature engineering and transformation pipelines
- `src/models/` — model training, evaluation, and persistence
- `app/` — serving layer (FastAPI and/or Streamlit)
- `notebooks/` — exploratory analysis; not part of the installable package
- `data/` — raw and processed datasets (gitignored, not committed)
- `models/` — serialized model artifacts (gitignored, not committed)
- `tests/` — pytest test suite

## Key Dependencies

| Library | Purpose |
|---|---|
| scikit-learn, xgboost, lightgbm | Model training |
| pandas, plotly | Data manipulation and visualization |
| mlflow | Experiment tracking and model registry |
| great-expectations | Data validation |
| fastapi + uvicorn | Model serving API |
| streamlit | Interactive data app |
