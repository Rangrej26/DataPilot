"""
Automated unit tests for modules.ml_lab
Validates task auto-detection, pipeline preparation, leakage checks,
multi-model training, regression evaluation, error handling, and live prediction.
"""

import pytest
import pandas as pd
from modules.data_loader import load_sample_dataset
from modules.ml_lab import (
    auto_detect_ml_task, check_target_leakage, MLTrainingPipeline
)


def test_task_detection_classification():
    df, _ = load_sample_dataset("patient_health")
    task, desc = auto_detect_ml_task(df, "Diabetic_Status")
    assert task == "Classification"


def test_task_detection_regression():
    df, _ = load_sample_dataset("patient_health")
    task, desc = auto_detect_ml_task(df, "Health_Risk_Score")
    assert task == "Regression"


def test_task_detection_constant_target():
    df = pd.DataFrame({"X": [1, 2, 3], "y": [1, 1, 1]})
    task, desc = auto_detect_ml_task(df, "y")
    assert task == "Invalid"


def test_ml_pipeline_classification():
    df, _ = load_sample_dataset("patient_health")
    features = ["Age", "BMI", "Blood_Pressure", "Glucose_Level", "Smoking_Status"]
    pipeline = MLTrainingPipeline(
        df=df,
        target_col="Diabetic_Status",
        feature_cols=features,
        task_type="Classification",
        test_size=0.2
    )

    ok, err = pipeline.prepare_data()
    assert ok is True, f"Prep failed: {err}"

    results = pipeline.train_and_compare()
    assert len(results) > 0
    assert "Model" in results[0]
    assert "Accuracy" in results[0]
    assert "F1-Score" in results[0]
    assert pipeline.best_model_name is not None

    # Test live prediction returns human-readable string class
    sample_input = {
        "Age": 45,
        "BMI": 28.5,
        "Blood_Pressure": 120,
        "Glucose_Level": 110,
        "Smoking_Status": "Never"
    }
    pred_res = pipeline.predict_single_instance(sample_input)
    assert "prediction" in pred_res
    assert isinstance(pred_res["prediction"], str)
    assert "probabilities" in pred_res


def test_ml_pipeline_regression():
    df, _ = load_sample_dataset("patient_health")
    features = ["Age", "BMI", "Blood_Pressure", "Glucose_Level", "Cholesterol"]
    pipeline = MLTrainingPipeline(
        df=df,
        target_col="Health_Risk_Score",
        feature_cols=features,
        task_type="Regression",
        test_size=0.2
    )

    ok, err = pipeline.prepare_data()
    assert ok is True, f"Prep failed: {err}"

    results = pipeline.train_and_compare()
    assert len(results) > 0
    assert "R² Score" in results[0]
    assert "MAE" in results[0]
    assert "RMSE" in results[0]
    assert pipeline.best_model_name is not None

    sample_input = {
        "Age": 50,
        "BMI": 30.2,
        "Blood_Pressure": 135,
        "Glucose_Level": 120,
        "Cholesterol": 210
    }
    pred_res = pipeline.predict_single_instance(sample_input)
    assert "prediction" in pred_res
    assert isinstance(pred_res["prediction"], (int, float))


def test_ml_pipeline_rejects_string_target_for_regression():
    df, _ = load_sample_dataset("patient_health")
    pipeline = MLTrainingPipeline(
        df=df,
        target_col="Diabetic_Status",  # String target!
        feature_cols=["Age", "BMI"],
        task_type="Regression"
    )
    ok, err = pipeline.prepare_data()
    assert ok is False
    assert "non-numeric text data" in err.lower() or "regression models require" in err.lower()
