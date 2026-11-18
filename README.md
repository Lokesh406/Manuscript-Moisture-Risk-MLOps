
# Predicting Moisture Damage in Rare Manuscript Storage Rooms

## Project Overview

This project implements an MLOps pipeline to predict moisture-related damage risk in historical manuscript storage rooms.

The system predicts whether environmental conditions indicate potential moisture risk within the next 48 hours.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- SQLite
- FastAPI
- Pydantic
- Pytest
- GitHub Actions

## Dataset

Two synthetic dataset versions are provided.

- Version 1: Normal storage conditions.
- Version 2: Monsoon storage conditions.

## Machine Learning

The project compares:

- Decision Tree Classifier
- Random Forest Classifier

Recall is prioritized because missing moisture risk can damage historical manuscripts.

## Feature Engineering

- Dew point temperature
- Condensation risk index
- Ventilation effectiveness ratio

## Model API

GET /health

POST /predict

## Testing

Run:

python -m pytest tests/test_api.py -v

## Run API

pip install -r requirements.txt

uvicorn main:app --reload

## Important Limitation

The dataset and 48-hour target labels are synthetic. Model performance does not establish real-world museum conservation accuracy.
