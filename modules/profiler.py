"""
DataPilot AI - Data Profiler & Exploratory Data Analysis (EDA)
Generates comprehensive statistical profiles, distribution visuals,
correlation matrices, and missingness analyses using Plotly.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_components import get_theme_colors, get_plotly_theme


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes extended descriptive statistics for numerical columns:
    Mean, Std, Min, 25%, Median, 75%, Max, Skewness, Kurtosis, Zeros, Outliers.
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()

    stats_list = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())
        zero_count = int((series == 0).sum())
        
        stats_list.append({
            "Column": col,
            "Count": int(len(series)),
            "Missing": int(df[col].isnull().sum()),
            "Mean": round(float(series.mean()), 2),
            "Std Dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "Min": round(float(series.min()), 2),
            "25% (Q1)": round(float(q1), 2),
            "Median": round(float(series.median()), 2),
            "75% (Q3)": round(float(q3), 2),
            "Max": round(float(series.max()), 2),
            "Skewness": round(float(series.skew()), 2) if len(series) > 2 else 0.0,
            "Zeros": zero_count,
            "Outliers (IQR)": outlier_count
        })

    return pd.DataFrame(stats_list)


def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes frequency and cardinality metrics for categorical / text columns.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()

    stats_list = []
    total_rows = len(df)
    
    for col in cat_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
            
        unique_cnt = series.nunique()
        top_val = series.mode().iloc[0] if not series.empty else "N/A"
        top_freq = int((series == top_val).sum()) if not series.empty else 0
        top_pct = round((top_freq / total_rows) * 100, 1) if total_rows > 0 else 0
        null_count = int(df[col].isnull().sum())

        stats_list.append({
            "Column": col,
            "Unique Values": unique_cnt,
            "Missing": null_count,
            "Top Category": top_val,
            "Top Frequency": top_freq,
            "Top Category %": f"{top_pct}%",
            "Cardinality": "High" if unique_cnt > 50 else ("Medium" if unique_cnt > 10 else "Low")
        })

    return pd.DataFrame(stats_list)


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson", theme_name: str = "Midnight Dark") -> Optional[go.Figure]:
    """
    Generates an interactive Plotly correlation heatmap for numerical features.
    """
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr(method=method).round(2)
    template = get_plotly_theme(theme_name)
    colors = get_theme_colors(theme_name)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[[0.0, "#222225"], [0.5, "#FFFFFF"], [1.0, "#B9ADF5"]],
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Inter, sans-serif", "color": "#202124"},
        colorbar=dict(title=f"{method.capitalize()} r")
    ))

    fig.update_layout(
        title=f"Feature Correlation Heatmap ({method.capitalize()} Method)",
        template=template,
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickangle=-45)
    )
    return fig


def plot_missing_values(df: pd.DataFrame, theme_name: str = "Midnight Dark") -> Optional[go.Figure]:
    """
    Visualizes missing value counts and percentages across all columns.
    """
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    
    if len(null_cols) == 0:
        return None

    pcts = (null_cols / len(df) * 100).round(1)
    plot_df = pd.DataFrame({
        "Column": null_cols.index,
        "Missing Count": null_cols.values,
        "Missing %": pcts.values
    }).sort_values("Missing Count", ascending=True)

    template = get_plotly_theme(theme_name)
    colors = get_theme_colors(theme_name)

    fig = px.bar(
        plot_df,
        x="Missing Count",
        y="Column",
        orientation="h",
        text="Missing %",
        color="Missing %",
        color_continuous_scale="Reds",
        template=template,
        title="Missing Values by Column (Count and %)"
    )
    
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        height=max(300, len(plot_df) * 35 + 80),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_column_distribution(df: pd.DataFrame, column: str, theme_name: str = "Midnight Dark") -> Optional[go.Figure]:
    """
    Plots the distribution of a single column:
    - Histogram with Box plot for numerical
    - Frequency bar chart for categorical
    """
    if column not in df.columns:
        return None

    series = df[column].dropna()
    template = get_plotly_theme(theme_name)
    colors = get_theme_colors(theme_name)

    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(
            df,
            x=column,
            marginal="box",
            nbins=35,
            template=template,
            color_discrete_sequence=[colors[0]],
            title=f"Distribution & Box Plot of '{column}'"
        )
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
        return fig
    else:
        counts = series.value_counts().head(15).reset_index()
        counts.columns = [column, "Frequency"]
        fig = px.bar(
            counts,
            x=column,
            y="Frequency",
            text="Frequency",
            template=template,
            color_discrete_sequence=[colors[1] if len(colors) > 1 else colors[0]],
            title=f"Top Category Frequencies for '{column}'"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40), xaxis=dict(tickangle=-30))
        return fig
