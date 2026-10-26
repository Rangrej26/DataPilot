"""
Automated unit tests for modules.cleaning_engine
"""

import pytest
import pandas as pd
from modules.data_loader import load_sample_dataset
from modules.cleaning_engine import (
    CleaningSession, detect_cleaning_issues, apply_remove_duplicates,
    apply_missing_imputation, apply_text_standardization,
    apply_date_parsing, apply_outlier_handling, apply_drop_column
)

@pytest.fixture
def ecommerce_df():
    df, _ = load_sample_dataset("ecommerce_churn")
    return df

def test_detect_cleaning_issues(ecommerce_df):
    issues = detect_cleaning_issues(ecommerce_df)
    assert len(issues) > 0
    issue_names = [i["name"] for i in issues]
    assert any("Duplicate" in name for name in issue_names)
    assert any("Missing" in name for name in issue_names)
    
    # Verify concise recommendation fields
    first_issue = issues[0]
    assert "problem" in first_issue
    assert "recommended_action" in first_issue
    assert "reason" in first_issue
    assert "preview" in first_issue

def test_remove_duplicates(ecommerce_df):
    session = CleaningSession(ecommerce_df)
    initial_rows = len(session.current_df)
    
    success, msg = apply_remove_duplicates(session)
    assert success is True
    assert len(session.current_df) < initial_rows
    assert len(session.audit_trail) == 1
    assert session.can_undo() is True

    # Test 3-part explanation
    entry = session.audit_trail[0]
    three_part = entry.get_three_part_summary()
    assert "found" in three_part and "duplicate" in three_part["found"].lower()
    assert "action" in three_part and "removed" in three_part["action"].lower()
    assert "why" in three_part

    # Test compact summary
    summary_bullets = session.get_compact_summary()
    assert len(summary_bullets) == 1
    assert "duplicate" in summary_bullets[0].lower()

    # Test Undo
    undo_ok, undo_msg = session.undo_last_step()
    assert undo_ok is True
    assert len(session.current_df) == initial_rows
    assert len(session.audit_trail) == 0

def test_impute_missing_values(ecommerce_df):
    session = CleaningSession(ecommerce_df)
    assert session.current_df["Annual_Income"].isnull().sum() > 0
    
    success, msg = apply_missing_imputation(session, column="Annual_Income", strategy="median")
    assert success is True
    assert session.current_df["Annual_Income"].isnull().sum() == 0
    assert len(session.audit_trail) == 1

    entry = session.audit_trail[0]
    assert "median" in entry.why.lower()
    assert "filled" in entry.what_changed.lower()

def test_text_standardization(ecommerce_df):
    session = CleaningSession(ecommerce_df)
    success, msg = apply_text_standardization(session, column="City", operation="trim_and_title")
    assert success is True
    for val in session.current_df["City"].dropna():
        assert val == val.strip()

def test_outlier_handling(ecommerce_df):
    session = CleaningSession(ecommerce_df)
    success, msg = apply_outlier_handling(session, column="Annual_Income", method="iqr", action="cap")
    assert success is True
    assert len(session.audit_trail) == 1
    assert session.current_df["Annual_Income"].max() < 1000000

def test_clean_dataset_no_issues():
    # Dataset with zero flaws
    clean_df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0],
        "C": ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]
    })
    issues = detect_cleaning_issues(clean_df)
    assert len(issues) == 0
