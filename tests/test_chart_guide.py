"""
Automated unit tests for modules.chart_guide
Validates chart recommendations, dataset inspection, chart generation,
and data-grounded insight derivation across multiple analytical goals.
"""

import pytest
import pandas as pd
from modules.data_loader import load_sample_dataset
from modules.chart_guide import (
    ANALYTICAL_GOALS, inspect_dataset_for_visualization,
    get_chart_recommendations, generate_guided_chart, derive_chart_insights
)


def test_analytical_goals_defined():
    assert "compare_categories" in ANALYTICAL_GOALS
    assert "time_trends" in ANALYTICAL_GOALS
    assert "distributions" in ANALYTICAL_GOALS
    assert "relationships" in ANALYTICAL_GOALS
    assert "composition" in ANALYTICAL_GOALS
    assert "data_quality" in ANALYTICAL_GOALS


def test_dataset_inspection():
    df, _ = load_sample_dataset("patient_health")
    inspection = inspect_dataset_for_visualization(df)
    assert len(inspection["num_cols"]) > 0
    assert len(inspection["cat_cols"]) > 0
    assert inspection["total_rows"] == 1200


def test_recommend_compare_categories():
    df, _ = load_sample_dataset("patient_health")
    rec = get_chart_recommendations(df, "compare_categories")
    assert rec["status"] == "ok"
    assert len(rec["recommendations"]) > 0
    chart_types = [r["chart_type"] for r in rec["recommendations"]]
    assert any("Bar" in ct for ct in chart_types)


def test_recommend_distributions():
    df, _ = load_sample_dataset("patient_health")
    rec = get_chart_recommendations(df, "distributions")
    assert rec["status"] == "ok"
    chart_types = [r["chart_type"] for r in rec["recommendations"]]
    assert "Histogram" in chart_types
    assert "Box Plot" in chart_types


def test_recommend_relationships():
    df, _ = load_sample_dataset("patient_health")
    rec = get_chart_recommendations(df, "relationships")
    assert rec["status"] == "ok"
    chart_types = [r["chart_type"] for r in rec["recommendations"]]
    assert "Scatter Plot" in chart_types


def test_recommend_composition():
    df, _ = load_sample_dataset("patient_health")
    rec = get_chart_recommendations(df, "composition")
    assert rec["status"] == "ok"
    chart_types = [r["chart_type"] for r in rec["recommendations"]]
    assert any("Donut" in ct or "Bar" in ct for ct in chart_types)


def test_generate_bar_chart():
    df, _ = load_sample_dataset("patient_health")
    fig, err = generate_guided_chart(
        df=df,
        chart_type="Bar Chart",
        x_col="Smoking_Status",
        y_col="BMI",
        agg_func="mean"
    )
    assert err is None
    assert fig is not None


def test_generate_scatter_chart():
    df, _ = load_sample_dataset("patient_health")
    fig, err = generate_guided_chart(
        df=df,
        chart_type="Scatter Plot",
        x_col="Age",
        y_col="Blood_Pressure"
    )
    assert err is None
    assert fig is not None


def test_derive_chart_insights():
    df, _ = load_sample_dataset("patient_health")
    insights = derive_chart_insights(
        df=df,
        chart_type="Bar Chart",
        x_col="Smoking_Status",
        y_col="BMI",
        agg_func="mean"
    )
    assert len(insights) > 0
    assert any("Highest Category" in ins for ins in insights)
