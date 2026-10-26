"""
DataPilot AI - Unit Tests for Intelligent Dataset Understanding & Semantic Profiler
Tests dynamic inference, column dictionaries, suggestions, and user context across multiple structurally distinct datasets.
"""

import pytest
import pandas as pd
import numpy as np
from modules.dataset_explainer import (
    generate_dataset_overview,
    _infer_semantic_meaning,
    _sanitize_example_value,
    _detect_dataset_domain,
    _build_dynamic_exploration_goals
)


@pytest.fixture
def hr_dataset():
    """HR Employee Workforce Dataset."""
    return pd.DataFrame({
        "Employee ID": [101, 102, 103, 104, 105],
        "Department": ["Engineering", "Sales", "Engineering", "Marketing", "HR"],
        "Salary": [85000, 62000, 91000, 58000, 64000],
        "Joining Date": pd.to_datetime(["2020-01-15", "2021-03-20", "2019-11-01", "2022-07-10", "2020-05-18"]),
        "Experience": [5, 3, 8, 2, 4],
        "Attrition": ["No", "Yes", "No", "No", "Yes"]
    })


@pytest.fixture
def clinical_dataset():
    """Clinical Healthcare Dataset."""
    return pd.DataFrame({
        "Patient ID": ["P001", "P002", "P003", "P004"],
        "Age": [45, 62, 38, 55],
        "BMI": [24.5, 29.8, 22.1, 31.4],
        "Blood Pressure": [120, 142, 115, 138],
        "Glucose Level": [95, 140, 88, 125],
        "Health Risk": ["Low", "High", "Low", "Moderate"]
    })


@pytest.fixture
def ecommerce_dataset():
    """E-Commerce Retail Dataset."""
    return pd.DataFrame({
        "Customer ID": ["C10", "C20", "C30", "C40"],
        "Total Spend": [150.50, 420.00, 89.20, 1250.00],
        "Purchases": [3, 8, 1, 15],
        "Preferred Category": ["Electronics", "Apparel", "Books", "Home"],
        "Churn": [0, 1, 0, 0]
    })


@pytest.fixture
def ambiguous_dataset():
    """Synthetic dataset with unrecognized, arbitrary column names."""
    return pd.DataFrame({
        "col_alpha": [1.1, 2.2, 3.3, 4.4],
        "metric_omega": [100, 200, 300, 400],
        "xyz_feature": ["A", "B", "A", "C"],
        "secret_token": ["abc123secret", "def456secret", "ghi789secret", "jkl012secret"]
    })


def test_semantic_meaning_known_and_ambiguous():
    """Tests semantic inference matches expected rules and safely falls back."""
    s_salary = pd.Series([50000, 60000])
    meaning, is_ai = _infer_semantic_meaning("Salary", s_salary)
    assert is_ai is True
    assert "salary" in meaning.lower() or "compensation" in meaning.lower()

    s_dept = pd.Series(["Sales", "Eng"])
    meaning, is_ai = _infer_semantic_meaning("Department", s_dept)
    assert is_ai is True
    assert "department" in meaning.lower()

    # Ambiguous column should explicitly state fallback
    s_amb = pd.Series([10, 20, 30])
    meaning, is_ai = _infer_semantic_meaning("arbitrary_unknown_col", s_amb)
    assert is_ai is False
    assert meaning == "Meaning could not be confidently inferred"


def test_sanitize_example_value():
    """Tests that sensitive values like tokens and secrets are masked."""
    s_pwd = pd.Series(["my_secret_token", "another_token"])
    masked = _sanitize_example_value(s_pwd, "user_password")
    assert masked == "••••••••"

    s_name = pd.Series(["Engineering", "Marketing"])
    clean_val = _sanitize_example_value(s_name, "Department")
    assert clean_val == "Engineering"


def test_hr_dataset_overview(hr_dataset):
    """Tests overview generation on HR Workforce dataset."""
    overview = generate_dataset_overview(hr_dataset, file_name="employees.xlsx", sheet_name="Workforce")
    
    assert overview["domain"] == "Human Resources & Workforce"
    assert overview["total_rows"] == 5
    assert overview["total_cols"] == 6

    desc = overview["description"]
    # Must be inferred dynamically, starting with 'This dataset appears to contain...'
    assert "appears to contain" in desc
    assert "employee" in desc.lower()
    assert "workforce" in desc.lower() or "salary" in desc.lower() or "experience" in desc.lower()

    # Verify column dictionary
    col_names = [c["name"] for c in overview["column_dictionary"]]
    assert "Employee ID" in col_names
    assert "Department" in col_names
    assert "Salary" in col_names

    dept_entry = [c for c in overview["column_dictionary"] if c["name"] == "Department"][0]
    assert dept_entry["detected_type"] == "Text / Categorical"
    assert dept_entry["is_ai_inferred"] is True

    # Verify suggestions include predictive modeling for Attrition
    sug_titles = [s["title"].lower() for s in overview["suggestions"]]
    assert any("attrition" in t for t in sug_titles)


def test_clinical_dataset_overview(clinical_dataset):
    """Tests overview generation on Clinical Healthcare dataset."""
    overview = generate_dataset_overview(clinical_dataset, file_name="patient_vitals.csv")

    assert overview["domain"] == "Healthcare & Clinical"
    desc = overview["description"]
    assert "patient" in desc.lower() or "health" in desc.lower() or "medical" in desc.lower()

    # Verify suggestions include predictive modeling for Health Risk
    sug_titles = [s["title"].lower() for s in overview["suggestions"]]
    assert any("risk" in t for t in sug_titles)


def test_ecommerce_dataset_overview(ecommerce_dataset):
    """Tests overview generation on E-Commerce dataset."""
    overview = generate_dataset_overview(ecommerce_dataset, file_name="online_retail_churn.csv")

    assert overview["domain"] == "E-Commerce & Retail"
    desc = overview["description"]
    assert "customer" in desc.lower() or "commercial" in desc.lower() or "transaction" in desc.lower()

    # Verify suggestions include churn modeling
    sug_titles = [s["title"].lower() for s in overview["suggestions"]]
    assert any("churn" in t for t in sug_titles)


def test_descriptions_change_dynamically(hr_dataset, clinical_dataset, ecommerce_dataset):
    """Confirms that descriptions dynamically change according to the uploaded data."""
    ov_hr = generate_dataset_overview(hr_dataset, file_name="hr.csv")
    ov_med = generate_dataset_overview(clinical_dataset, file_name="med.csv")
    ov_ecom = generate_dataset_overview(ecommerce_dataset, file_name="shop.csv")

    # None of the descriptions should be identical
    assert ov_hr["description"] != ov_med["description"]
    assert ov_med["description"] != ov_ecom["description"]
    assert ov_hr["description"] != ov_ecom["description"]


def test_user_context_incorporation(ecommerce_dataset):
    """Tests that user-provided context is incorporated into the plain-English explanation."""
    ctx = "Monthly sales records for my college project"
    overview = generate_dataset_overview(ecommerce_dataset, file_name="sales.csv", user_context=ctx)
    
    assert ctx in overview["description"]
    assert "appears to contain" in overview["description"]
    # Underlying data structure must remain intact
    assert overview["total_rows"] == 4
    assert overview["total_cols"] == 5


def test_ambiguous_columns_fallback(ambiguous_dataset):
    """Tests that unknown column names cleanly return 'Meaning could not be confidently inferred'."""
    overview = generate_dataset_overview(ambiguous_dataset, file_name="custom_math.csv")

    col_dict = {c["name"]: c for c in overview["column_dictionary"]}
    assert col_dict["col_alpha"]["semantic_meaning"] == "Meaning could not be confidently inferred"
    assert col_dict["col_alpha"]["is_ai_inferred"] is False
    assert col_dict["metric_omega"]["semantic_meaning"] == "Meaning could not be confidently inferred"

    # Token should have masked example value
    assert col_dict["secret_token"]["example_value"] == "••••••••"
