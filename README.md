# AI Engineer Professional Suite

A production-ready AI engineering template designed with professional software development standards.

## Architecture

This project is structured into three main components:
1.  **Data Pipeline (`src/data_pipeline.py`)**: Responsible for robust data ingestion, cleaning, missing value imputation, and feature engineering using Pandas and Numpy.
2.  **Model Trainer (`src/model_trainer.py`)**: Handles the training lifecycle, including train/test splitting, hyperparameter tuning via `GridSearchCV`, evaluation, and artifact serialization.
3.  **Model API (`api/main.py`)**: A high-performance REST API built with FastAPI, using Pydantic for strict input validation, enabling reliable model serving.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MulaRamC32s/AI-Engineer-Professional-Suite.git
    cd AI-Engineer-Professional-Suite
    ```

2.  **Create a virtual environment & install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

**1. Train the Model:**
Run the training script to generate the model artifact (`model.joblib`).
```bash
python -m src.model_trainer
```

**2. Run the API:**
Start the FastAPI server using Uvicorn.
```bash
uvicorn api.main:app --reload
```
Access the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Features
*   **Modular Design**: Clear separation of concerns between data processing, modeling, and serving.
*   **Production-Ready API**: FastAPI integration with built-in Swagger UI and Pydantic validation schemas.
*   **Containerized**: Includes a multi-stage Dockerfile optimized for Python ML applications.
*   **Automated Tuning**: Built-in hyperparameter tuning for optimal model performance.
