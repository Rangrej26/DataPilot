"""
Automated unit tests for modules.sql_engine
"""

import pytest
import pandas as pd
from modules.data_loader import load_sample_dataset
from modules.sql_engine import (
    SQLExecutionEngine, validate_sql_security, explain_sql_query,
    generate_suggested_queries, rule_based_nl_to_sql
)

@pytest.fixture
def sample_df():
    df, _ = load_sample_dataset("patient_health")
    return df

def test_sql_execution(sample_df):
    engine = SQLExecutionEngine()
    engine.register_table("patient_data", sample_df)
    
    query = "SELECT COUNT(*) AS total_pts, ROUND(AVG(Age), 1) AS avg_age FROM patient_data;"
    res_df, runtime, err = engine.execute_query(query)
    
    assert err is None
    assert res_df is not None
    assert len(res_df) == 1
    assert "total_pts" in res_df.columns
    assert res_df.iloc[0]["total_pts"] == len(sample_df)

def test_sql_security_sandbox():
    engine = SQLExecutionEngine()
    
    # Dangerous statements must fail validation
    dangerous_queries = [
        "DROP TABLE data;",
        "DELETE FROM data WHERE 1=1;",
        "INSERT INTO data VALUES (1, 2);",
        "SELECT * FROM read_csv('secret.csv');",
        "COPY data TO 'output.csv';",
        "SELECT 1; SELECT 2;"
    ]
    
    for dq in dangerous_queries:
        is_safe, msg = validate_sql_security(dq)
        assert is_safe is False, f"Expected unsafe for: {dq}"
        res_df, _, err = engine.execute_query(dq)
        assert err is not None

def test_sql_explainer():
    query = "SELECT City, AVG(Annual_Income) FROM data WHERE Age > 25 GROUP BY City ORDER BY 2 DESC LIMIT 5;"
    explanation = explain_sql_query(query)
    assert "SELECT Clause" in explanation
    assert "WHERE Clause" in explanation
    assert "GROUP BY Clause" in explanation
    assert "ORDER BY Clause" in explanation

def test_nl_to_sql(sample_df):
    sql = rule_based_nl_to_sql("What is the top 5 records by age?", sample_df)
    assert "SELECT" in sql
    assert "ORDER BY" in sql
    assert "LIMIT 5" in sql
