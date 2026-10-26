"""
DataPilot AI - Beginner-Friendly Chart Guide & Visualization Assistant
Analyzes dataset schema, data types, and cardinality to recommend suitable charts 
based on the user's analytical goal. Provides plain-English axis explanations, 
generates interactive Plotly visualizations, and produces data-grounded interpretations.
Operates 100% deterministically without external LLMs.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_components import get_theme_colors, get_plotly_theme


# Supported analytical goals
ANALYTICAL_GOALS = {
    "compare_categories": {
        "title": "Compare Categories",
        "description": "Compare metrics across distinct groups, departments, or product categories.",
        "icon": "📊",
        "example": "Which department has the highest average salary?"
    },
    "time_trends": {
        "title": "See Changes Over Time",
        "description": "Track historical trajectories, seasonality, and chronological patterns.",
        "icon": "📈",
        "example": "How has total revenue changed month over month?"
    },
    "distributions": {
        "title": "Understand Distributions & Spread",
        "description": "Examine distribution shapes, central tendency, spread, and outlier locations.",
        "icon": "📉",
        "example": "What is the age spread of our patients, and are there extreme outliers?"
    },
    "relationships": {
        "title": "Explore Relationships Between Numerical Variables",
        "description": "Evaluate correlations, co-variations, and scatter patterns between two numbers.",
        "icon": "🔗",
        "example": "Does higher physical activity correlate with lower blood pressure?"
    },
    "composition": {
        "title": "Compare Proportions or Composition",
        "description": "Evaluate how distinct parts contribute to a whole total.",
        "icon": "🥧",
        "example": "What percentage of overall sales is driven by each product category?"
    },
    "data_quality": {
        "title": "Examine Missing Values & Data Quality",
        "description": "Identify data completeness, missing value distribution, and quality gaps.",
        "icon": "❓",
        "example": "Which columns have missing data and require cleaning?"
    }
}

# Backward compatibility alias
CHART_GOALS = ANALYTICAL_GOALS


def recommend_charts_for_goal(
    goal_key_or_df: Any,
    schema_info_or_goal: Any = None
) -> List[Dict[str, Any]]:
    """Universal backward-compatible adapter for chart recommendations returning a list of recommendation dicts."""
    if isinstance(goal_key_or_df, pd.DataFrame):
        df = goal_key_or_df
        goal_key = str(schema_info_or_goal) if schema_info_or_goal else "compare_categories"
    else:
        goal_key = str(goal_key_or_df)
        df = None

    if df is not None:
        rec_data = get_chart_recommendations(df, goal_key)
        return rec_data.get("recommendations", [])
    
    from modules.data_loader import load_sample_dataset
    dummy_df, _ = load_sample_dataset("patient_health")
    rec_data = get_chart_recommendations(dummy_df, goal_key)
    return rec_data.get("recommendations", [])


def inspect_dataset_for_visualization(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects column types, cardinality, and data suitability for charting.
    """
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string", "bool"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)

    # Detect string columns that look like dates
    potential_dates = []
    for c in cat_cols:
        if any(k in c.lower() for k in ["date", "time", "year", "month", "timestamp"]):
            potential_dates.append(c)

    # Cardinality dictionary
    cardinality = {c: int(df[c].nunique()) for c in df.columns}

    # Low-cardinality categories suitable for X-axis / grouping (< 35 unique values)
    low_card_cats = [c for c in cat_cols if 1 < cardinality.get(c, 0) <= 35]
    very_low_card_cats = [c for c in cat_cols if 1 < cardinality.get(c, 0) <= 8]

    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "date_cols": date_cols,
        "potential_dates": potential_dates,
        "low_card_cats": low_card_cats,
        "very_low_card_cats": very_low_card_cats,
        "cardinality": cardinality
    }


def get_chart_recommendations(df: pd.DataFrame, goal_key: str) -> Dict[str, Any]:
    """
    Recommends suitable charts tailored specifically to the uploaded dataset and selected goal.
    Provides beginner-friendly explanations of why the chart fits, required columns, and axis roles.
    """
    inspection = inspect_dataset_for_visualization(df)
    num_cols = inspection["num_cols"]
    cat_cols = inspection["cat_cols"]
    date_cols = inspection["date_cols"]
    low_card_cats = inspection["low_card_cats"]
    very_low_card_cats = inspection["very_low_card_cats"]

    recommendations = []
    goal_info = ANALYTICAL_GOALS.get(goal_key, ANALYTICAL_GOALS["compare_categories"])

    # -----------------------------------------------------------------------
    # GOAL 1: COMPARE CATEGORIES
    # -----------------------------------------------------------------------
    if goal_key == "compare_categories":
        if low_card_cats and num_cols:
            x_candidate = low_card_cats[0]
            y_candidate = num_cols[0]
            recommendations.append({
                "chart_type": "Bar Chart",
                "badge": "Highly Recommended",
                "suggested_x": x_candidate,
                "suggested_y": y_candidate,
                "suggested_agg": "mean",
                "suggested_color": None,
                "why_useful": f"A Bar Chart makes it effortless to visually rank and contrast '{y_candidate}' across different '{x_candidate}' segments.",
                "columns_needed": "1 Categorical feature (X-axis) and 1 Numerical metric (Y-axis with aggregation).",
                "axes_explanation": f"• X-Axis: Represents distinct categories in '{x_candidate}'.\n• Y-Axis: Represents average or total '{y_candidate}'.",
                "what_to_look_for": "Identify which category has the tallest bar (highest value) and lowest bar, and notice whether variations are dramatic or uniform.",
                "when_to_avoid": "Avoid when your category has more than 40 unique values, as bar labels will overlap and become unreadable."
            })

            if len(low_card_cats) >= 2:
                recommendations.append({
                    "chart_type": "Grouped Bar Chart",
                    "badge": "Recommended for Sub-groups",
                    "suggested_x": low_card_cats[0],
                    "suggested_y": y_candidate,
                    "suggested_agg": "sum",
                    "suggested_color": low_card_cats[1],
                    "why_useful": f"Allows multi-dimensional comparison of '{y_candidate}' across '{low_card_cats[0]}' segmented by '{low_card_cats[1]}'.",
                    "columns_needed": "2 Categorical features (primary category + grouping color) and 1 Numerical metric.",
                    "axes_explanation": f"• X-Axis: Primary category '{low_card_cats[0]}'.\n• Color: Sub-group '{low_card_cats[1]}'.\n• Y-Axis: Metric '{y_candidate}'.",
                    "what_to_look_for": "Look for interactions—e.g., does one sub-group dominate within specific main categories?",
                    "when_to_avoid": "Avoid if both categories have many values, resulting in overcrowded visual clusters."
                })
        elif low_card_cats:
            recommendations.append({
                "chart_type": "Bar Chart (Count)",
                "badge": "Recommended",
                "suggested_x": low_card_cats[0],
                "suggested_y": None,
                "suggested_agg": "count",
                "suggested_color": None,
                "why_useful": f"Shows the record frequency distribution across '{low_card_cats[0]}'.",
                "columns_needed": "1 Categorical feature.",
                "axes_explanation": f"• X-Axis: Categories in '{low_card_cats[0]}'.\n• Y-Axis: Count of records in each category.",
                "what_to_look_for": "Check if one category contains a disproportionate share of total records.",
                "when_to_avoid": "Not suitable for measuring monetary or continuous values directly."
            })
        else:
            return {
                "goal": goal_info,
                "status": "warning",
                "message": "Category comparison requires at least one categorical feature with moderate cardinality. This dataset currently lacks suitable categorical columns.",
                "recommendations": []
            }

    # -----------------------------------------------------------------------
    # GOAL 2: TIME TRENDS
    # -----------------------------------------------------------------------
    elif goal_key == "time_trends":
        effective_date = date_cols[0] if date_cols else (inspection["potential_dates"][0] if inspection["potential_dates"] else None)
        
        if effective_date and num_cols:
            y_cand = num_cols[0]
            recommendations.append({
                "chart_type": "Line Chart",
                "badge": "Highly Recommended",
                "suggested_x": effective_date,
                "suggested_y": y_cand,
                "suggested_agg": "mean",
                "suggested_color": None,
                "why_useful": f"A Line Chart connects chronological data points to clearly reveal upward trajectories, downward dips, and cyclical cycles in '{y_cand}'.",
                "columns_needed": "1 Date/Time feature (X-axis) and 1 Numerical metric (Y-axis).",
                "axes_explanation": f"• X-Axis: Chronological dates in '{effective_date}'.\n• Y-Axis: Metric '{y_cand}'.",
                "what_to_look_for": "Identify overall direction (upward growth, plateau, or decline) and sharp temporary spikes or drops.",
                "when_to_avoid": "Avoid if X is unordered categorical data or if timestamps are completely random."
            })
        else:
            msg = "Time-series trend analysis requires a valid date or timestamp column. "
            if inspection["potential_dates"]:
                msg += f"Found potential date columns: {inspection['potential_dates']}. Use 'Smart Data Cleaning' to parse them into datetime format first."
            else:
                msg += "No temporal columns were detected in this dataset."
            return {
                "goal": goal_info,
                "status": "warning",
                "message": msg,
                "recommendations": []
            }

    # -----------------------------------------------------------------------
    # GOAL 3: DISTRIBUTIONS & SPREAD
    # -----------------------------------------------------------------------
    elif goal_key == "distributions":
        if num_cols:
            num_cand = num_cols[0]
            recommendations.append({
                "chart_type": "Histogram",
                "badge": "Highly Recommended",
                "suggested_x": num_cand,
                "suggested_y": None,
                "suggested_agg": None,
                "suggested_color": None,
                "why_useful": f"A Histogram bins '{num_cand}' into intervals to reveal the shape of distribution (bell curve, right-skewed, bimodal).",
                "columns_needed": "1 Continuous numerical feature.",
                "axes_explanation": f"• X-Axis: Binned value ranges of '{num_cand}'.\n• Y-Axis: Count of rows falling into each bin.",
                "what_to_look_for": "Look for symmetry vs skewness, single peak vs multiple peaks, and values isolated far from the main body.",
                "when_to_avoid": "Not suitable for discrete text labels or non-numeric identifiers."
            })

            recommendations.append({
                "chart_type": "Box Plot",
                "badge": "Recommended for Outliers",
                "suggested_x": low_card_cats[0] if low_card_cats else None,
                "suggested_y": num_cand,
                "suggested_agg": None,
                "suggested_color": None,
                "why_useful": f"A Box Plot summarizes the median, 25th/75th quartiles, and flags statistical outliers in '{num_cand}' as distinct points.",
                "columns_needed": "1 Numerical metric (Y-axis) with optional grouping Category (X-axis).",
                "axes_explanation": f"• Box center: Median value.\n• Box bounds: Interquartile Range (IQR, 25% to 75%).\n• Points beyond whiskers: Statistical outliers.",
                "what_to_look_for": "Identify whether outliers cluster high or low, and compare box sizes to evaluate variance across categories.",
                "when_to_avoid": "Less intuitive for non-technical audiences compared to a standard bar chart."
            })
        else:
            return {
                "goal": goal_info,
                "status": "warning",
                "message": "Distribution analysis requires numerical features. None were found in this dataset.",
                "recommendations": []
            }

    # -----------------------------------------------------------------------
    # GOAL 4: NUMERICAL RELATIONSHIPS (CORRELATION)
    # -----------------------------------------------------------------------
    elif goal_key == "relationships":
        if len(num_cols) >= 2:
            x_cand, y_cand = num_cols[0], num_cols[1]
            recommendations.append({
                "chart_type": "Scatter Plot",
                "badge": "Highly Recommended",
                "suggested_x": x_cand,
                "suggested_y": y_cand,
                "suggested_agg": None,
                "suggested_color": very_low_card_cats[0] if very_low_card_cats else None,
                "why_useful": f"A Scatter Plot plots each record as a point at coordinates ('{x_cand}', '{y_cand}') to show directional co-variation and clusters.",
                "columns_needed": "2 Numerical features (X and Y), with optional Categorical color coding.",
                "axes_explanation": f"• X-Axis: '{x_cand}'.\n• Y-Axis: '{y_cand}'.",
                "what_to_look_for": "Check if points slant upward (positive correlation), downward (negative correlation), or form a diffuse cloud (no correlation).",
                "when_to_avoid": "Avoid when either column has only 1 or 2 discrete integer values (use a grouped bar or box plot instead)."
            })

            if len(num_cols) >= 3:
                recommendations.append({
                    "chart_type": "Correlation Heatmap",
                    "badge": "Recommended for Overview",
                    "suggested_x": None,
                    "suggested_y": None,
                    "suggested_agg": None,
                    "suggested_color": None,
                    "why_useful": f"Computes Pearson correlation coefficients between all {len(num_cols)} numerical features simultaneously in an annotated grid.",
                    "columns_needed": "3 or more Numerical features.",
                    "axes_explanation": "Both axes represent numerical features; cell color indicates strength (-1.0 to +1.0).",
                    "what_to_look_for": "Look for dark blue or warm colored cells representing strong linear relationships.",
                    "when_to_avoid": "Does not reveal non-linear patterns or single-record outliers."
                })
        else:
            return {
                "goal": goal_info,
                "status": "warning",
                "message": f"Exploring relationships between variables requires at least two numerical columns. This dataset only contains {len(num_cols)} numerical feature(s).",
                "recommendations": []
            }

    # -----------------------------------------------------------------------
    # GOAL 5: COMPOSITION & PROPORTIONS
    # -----------------------------------------------------------------------
    elif goal_key == "composition":
        if very_low_card_cats:
            cat_cand = very_low_card_cats[0]
            val_cand = num_cols[0] if num_cols else None
            recommendations.append({
                "chart_type": "Donut Chart",
                "badge": "Recommended",
                "suggested_x": cat_cand,
                "suggested_y": val_cand,
                "suggested_agg": "sum" if val_cand else "count",
                "suggested_color": None,
                "why_useful": f"A Donut Chart visualizes relative percentage contribution of '{cat_cand}' segments to the whole total.",
                "columns_needed": f"1 Categorical feature with few classes ({inspection['cardinality'].get(cat_cand)} classes) and optional Numerical value.",
                "axes_explanation": f"Slices represent '{cat_cand}'; slice arc angle represents share of total {val_cand or 'record count'}.",
                "what_to_look_for": "Identify whether one segment represents more than 50% of the total, or if slices are evenly divided.",
                "when_to_avoid": "NEVER use when categories exceed 7, as human visual perception struggles to compare angles of small slices. Use a Bar Chart instead."
            })
        elif low_card_cats:
            recommendations.append({
                "chart_type": "Bar Chart (Proportions)",
                "badge": "Better Alternative to Pie Chart",
                "suggested_x": low_card_cats[0],
                "suggested_y": num_cols[0] if num_cols else None,
                "suggested_agg": "sum" if num_cols else "count",
                "suggested_color": None,
                "why_useful": f"Because '{low_card_cats[0]}' has {inspection['cardinality'].get(low_card_cats[0])} classes, a Bar Chart is significantly more readable than a crowded pie chart.",
                "columns_needed": "1 Categorical feature + 1 Numerical metric.",
                "axes_explanation": "X-axis represents categories; Y-axis shows comparative totals.",
                "what_to_look_for": "Compare relative height of bars to gauge dominance.",
                "when_to_avoid": "Avoid if exact whole-part percentage is required."
            })
        else:
            return {
                "goal": goal_info,
                "status": "warning",
                "message": "Composition charts require categorical columns with a small number of categories (2 to 7). None were found.",
                "recommendations": []
            }

    # -----------------------------------------------------------------------
    # GOAL 6: DATA QUALITY & COMPLETENESS
    # -----------------------------------------------------------------------
    elif goal_key == "data_quality":
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            recommendations.append({
                "chart_type": "Missingness Bar Chart",
                "badge": "Highly Recommended",
                "suggested_x": None,
                "suggested_y": None,
                "suggested_agg": None,
                "suggested_color": None,
                "why_useful": f"Ranks all {len(cols_with_nulls)} columns with missing cells by count and percentage.",
                "columns_needed": "Dataset columns containing missing values.",
                "axes_explanation": "Y-Axis: Columns with missing data. X-Axis: Missing value counts and percentages.",
                "what_to_look_for": "Look for columns where missing percentage exceeds 30%, which may require dropping or specialized imputation.",
                "when_to_avoid": "Not needed if the dataset is 100% complete."
            })
        else:
            return {
                "goal": goal_info,
                "status": "success",
                "message": "🎉 Outstanding! This dataset contains zero missing values across all columns. No data gaps detected.",
                "recommendations": []
            }

    return {
        "goal": goal_info,
        "status": "ok",
        "message": f"Found {len(recommendations)} recommended visualization(s) based on your goal and data structure.",
        "recommendations": recommendations
    }


def _generate_guided_chart_core(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    agg_func: str = "mean",
    color_col: Optional[str] = None,
    theme_name: str = "Professional Light"
) -> Tuple[Optional[go.Figure], Optional[str]]:
    """
    Renders an actual Plotly chart matching user parameters.
    Validates input variables and returns (Figure, error_message).
    """
    try:
        template = get_plotly_theme(theme_name)
        colors = get_theme_colors(theme_name)

        # -------------------------------------------------------------------
        # 1. Bar Chart
        # -------------------------------------------------------------------
        if chart_type in ["Bar Chart", "Bar Chart (Count)", "Bar Chart (Proportions)", "Grouped Bar Chart"]:
            if not x_col or x_col not in df.columns:
                return None, "Please select a valid column for the X-Axis."

            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                # Group by and aggregate
                group_cols = [x_col]
                if color_col and color_col in df.columns and color_col != x_col:
                    group_cols.append(color_col)

                if agg_func == "sum":
                    plot_data = df.groupby(group_cols)[y_col].sum().reset_index()
                    y_label = f"Total {y_col}"
                elif agg_func == "mean":
                    plot_data = df.groupby(group_cols)[y_col].mean().round(2).reset_index()
                    y_label = f"Average {y_col}"
                elif agg_func == "median":
                    plot_data = df.groupby(group_cols)[y_col].median().round(2).reset_index()
                    y_label = f"Median {y_col}"
                elif agg_func == "min":
                    plot_data = df.groupby(group_cols)[y_col].min().reset_index()
                    y_label = f"Minimum {y_col}"
                elif agg_func == "max":
                    plot_data = df.groupby(group_cols)[y_col].max().reset_index()
                    y_label = f"Maximum {y_col}"
                else:
                    plot_data = df.groupby(group_cols)[y_col].count().reset_index()
                    y_label = f"Count of {y_col}"

                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col in plot_data.columns else None,
                    barmode="group" if color_col in plot_data.columns else "relative",
                    template=template,
                    color_discrete_sequence=colors,
                    labels={y_col: y_label, x_col: x_col},
                    title=f"{y_label} across {x_col}"
                )
            else:
                # Count frequency
                plot_data = df[x_col].value_counts().reset_index()
                plot_data.columns = [x_col, "Count"]
                fig = px.bar(
                    plot_data.head(30),
                    x=x_col,
                    y="Count",
                    template=template,
                    color_discrete_sequence=[colors[0]],
                    title=f"Record Frequency by {x_col}"
                )

            fig.update_layout(xaxis_tickangle=-30, height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 2. Line Chart
        # -------------------------------------------------------------------
        elif chart_type == "Line Chart":
            if not x_col or not y_col:
                return None, "Line Chart requires both an X-Axis and a Y-Axis column."
            
            # Sort chronologically by X
            clean_df = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
            if color_col and color_col in clean_df.columns:
                plot_data = clean_df.groupby([x_col, color_col])[y_col].mean().reset_index()
            else:
                plot_data = clean_df.groupby(x_col)[y_col].mean().reset_index()

            fig = px.line(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_col if color_col in plot_data.columns else None,
                template=template,
                color_discrete_sequence=colors,
                title=f"{y_col} Trend over {x_col}",
                markers=True
            )
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 3. Histogram
        # -------------------------------------------------------------------
        elif chart_type == "Histogram":
            if not x_col:
                return None, "Histogram requires a numerical column for the X-Axis."
            
            clean_df = df.dropna(subset=[x_col])
            fig = px.histogram(
                clean_df,
                x=x_col,
                color=color_col if color_col and color_col in clean_df.columns else None,
                template=template,
                color_discrete_sequence=colors,
                nbins=30,
                marginal="box",
                title=f"Distribution of {x_col} with Marginal Box Plot"
            )
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 4. Box Plot
        # -------------------------------------------------------------------
        elif chart_type == "Box Plot":
            if not y_col:
                return None, "Box Plot requires a numerical column for the Y-Axis."
            
            clean_df = df.dropna(subset=[y_col])
            fig = px.box(
                clean_df,
                x=x_col if x_col and x_col in clean_df.columns else None,
                y=y_col,
                color=color_col if color_col and color_col in clean_df.columns else None,
                template=template,
                color_discrete_sequence=colors,
                points="outliers",
                title=f"Box Plot: Quartiles & Outliers in {y_col}" + (f" by {x_col}" if x_col else "")
            )
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 5. Scatter Plot
        # -------------------------------------------------------------------
        elif chart_type == "Scatter Plot":
            if not x_col or not y_col:
                return None, "Scatter Plot requires both an X-Axis and Y-Axis column."
            
            clean_df = df.dropna(subset=[x_col, y_col])
            fig = px.scatter(
                clean_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in clean_df.columns else None,
                template=template,
                color_discrete_sequence=colors,
                title=f"Scatter Plot: {y_col} vs. {x_col}"
            )
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 6. Donut Chart
        # -------------------------------------------------------------------
        elif chart_type == "Donut Chart":
            if not x_col:
                return None, "Donut Chart requires a categorical column."
            
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                plot_data = df.groupby(x_col)[y_col].sum().reset_index()
                val_col = y_col
            else:
                plot_data = df[x_col].value_counts().reset_index()
                plot_data.columns = [x_col, "Count"]
                val_col = "Count"

            fig = px.pie(
                plot_data.head(10),
                names=x_col,
                values=val_col,
                hole=0.45,
                template=template,
                color_discrete_sequence=colors,
                title=f"Composition Share of {x_col} by {val_col}"
            )
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        # -------------------------------------------------------------------
        # 7. Correlation Heatmap
        # -------------------------------------------------------------------
        elif chart_type == "Correlation Heatmap":
            num_df = df.select_dtypes(include=["number"])
            if num_df.shape[1] < 2:
                return None, "Correlation Heatmap requires at least 2 numerical columns."
            
            corr = num_df.corr().round(2)
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale=[[0.0, "#222225"], [0.5, "#FFFFFF"], [1.0, "#B9ADF5"]],
                zmin=-1, zmax=1,
                text=corr.values,
                texttemplate="%{text}",
                textfont={"size": 12, "family": "Inter, sans-serif", "color": "#202124"}
            ))
            fig.update_layout(
                title="Numerical Correlation Heatmap (Pearson r)",
                template=template,
                height=450,
                margin=dict(l=40, r=40, t=50, b=50)
            )
            return fig, None

        # -------------------------------------------------------------------
        # 8. Missingness Bar Chart
        # -------------------------------------------------------------------
        elif chart_type == "Missingness Bar Chart":
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            if len(null_cols) == 0:
                return None, "Dataset has zero missing values."
            
            plot_df = pd.DataFrame({
                "Feature": null_cols.index,
                "Missing Count": null_cols.values,
                "Missing %": (null_cols / len(df) * 100).round(1).values
            }).sort_values("Missing Count", ascending=True)

            fig = px.bar(
                plot_df,
                x="Missing Count",
                y="Feature",
                orientation="h",
                text="Missing %",
                color="Missing %",
                color_continuous_scale="Reds",
                template=template,
                title="Missing Value Distribution by Column"
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=max(320, len(plot_df) * 35 + 80), margin=dict(l=40, r=40, t=50, b=50))
            return fig, None

        else:
            return None, f"Unsupported chart type: {chart_type}"

    except Exception as e:
        return None, f"Chart generation error: {str(e)}"


class FlexibleChartResult:
    """Supports unpacking as 2 elements (fig, err) OR 3 elements (fig, data, err)."""
    def __init__(self, fig: Any, data: Any, err: Any):
        self.fig = fig
        self.data = data
        self.err = err

    def __iter__(self):
        try:
            import inspect
            import dis
            frame = inspect.currentframe().f_back
            instrs = list(dis.get_instructions(frame.f_code))
            for instr in instrs:
                if instr.offset >= frame.f_lasti:
                    if 'UNPACK' in instr.opname:
                        if instr.argval == 3:
                            return iter([self.fig, self.data, self.err])
                        else:
                            return iter([self.fig, self.err])
        except Exception:
            pass
        return iter([self.fig, self.err])

    def __getitem__(self, idx: int) -> Any:
        if idx == 0:
            return self.fig
        elif idx == 1:
            return self.err
        elif idx == 2:
            return self.data
        raise IndexError("Index out of range")


def generate_guided_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    agg_func: str = "mean",
    color_col: Optional[str] = None,
    theme_name: str = "Professional Light",
    aggregation: Optional[str] = None,
    top_n: Optional[int] = None,
    **kwargs: Any
) -> Any:
    """
    Renders an actual Plotly chart matching user parameters.
    Returns FlexibleChartResult that cleanly unpacks into 2 items (fig, err) OR 3 items (fig, data, err).
    """
    if aggregation is not None:
        agg_func = aggregation

    fig, err = _generate_guided_chart_core(
        df=df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        agg_func=agg_func,
        color_col=color_col,
        theme_name=theme_name
    )
    return FlexibleChartResult(fig, df, err)


def derive_chart_insights(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    agg_func: str = "mean",
    color_col: Optional[str] = None
) -> List[str]:
    """
    Computes deterministic, data-grounded interpretations derived directly from the plotted figures.
    Never hallucinates or invents statistics.
    """
    insights = []
    try:
        # Category comparison insights
        if chart_type in ["Bar Chart", "Bar Chart (Count)", "Bar Chart (Proportions)", "Grouped Bar Chart"]:
            if x_col and y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                if agg_func == "sum":
                    grp = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    stat_name = "total"
                else:
                    grp = df.groupby(x_col)[y_col].mean().round(2).sort_values(ascending=False)
                    stat_name = "average"

                if not grp.empty:
                    top_cat, top_val = grp.index[0], grp.iloc[0]
                    bot_cat, bot_val = grp.index[-1], grp.iloc[-1]
                    overall_avg = round(float(df[y_col].mean()), 2)
                    
                    insights.append(f"**Highest Category:** '{top_cat}' has the highest {stat_name} {y_col} at **{top_val:,}**.")
                    insights.append(f"**Lowest Category:** '{bot_cat}' has the lowest {stat_name} {y_col} at **{bot_val:,}**.")
                    if overall_avg > 0:
                        diff_pct = round(((top_val - overall_avg) / overall_avg) * 100, 1)
                        if diff_pct > 0:
                            insights.append(f"**Benchmark:** '{top_cat}' is **{diff_pct}%** above the overall dataset average ({overall_avg:,}).")
            elif x_col:
                counts = df[x_col].value_counts()
                if not counts.empty:
                    top_cat, top_cnt = counts.index[0], counts.iloc[0]
                    top_pct = round((top_cnt / len(df)) * 100, 1)
                    insights.append(f"**Dominant Category:** '{top_cat}' represents **{top_pct}%** of all records ({top_cnt:,} rows).")

        # Numerical distribution insights
        elif chart_type in ["Histogram", "Box Plot"] and x_col and pd.api.types.is_numeric_dtype(df[x_col]):
            series = df[x_col].dropna()
            if not series.empty:
                med = round(float(series.median()), 2)
                mean = round(float(series.mean()), 2)
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())
                skew = round(float(series.skew()), 2)

                insights.append(f"**Central Tendency:** Median {x_col} is **{med:,}** (Mean is {mean:,}).")
                skew_desc = "approximately symmetric" if abs(skew) < 0.5 else ("right-skewed (high values tail)" if skew > 0 else "left-skewed")
                insights.append(f"**Shape:** The distribution is **{skew_desc}** (Skewness = {skew}).")
                if outliers > 0:
                    insights.append(f"**Outliers:** Found **{outliers:,} statistical outlier(s)** beyond 1.5× IQR fences.")
                else:
                    insights.append("**Outliers:** Zero statistical outliers detected in this feature.")

        # Scatter relationship insights
        elif chart_type == "Scatter Plot" and x_col and y_col and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            valid_df = df.dropna(subset=[x_col, y_col])
            if len(valid_df) > 2:
                r = round(float(valid_df[x_col].corr(valid_df[y_col])), 3)
                strength = "strong" if abs(r) >= 0.7 else ("moderate" if abs(r) >= 0.35 else "weak")
                direction = "positive" if r > 0 else "negative"
                insights.append(f"**Correlation Coefficient:** Pearson r = **{r}** ({strength} {direction} association).")
                if abs(r) >= 0.35:
                    insights.append(f"**Trend:** As '{x_col}' increases, '{y_col}' tends to **{'increase' if r > 0 else 'decrease'}**.")
                else:
                    insights.append(f"**Trend:** There is little linear relationship between '{x_col}' and '{y_col}'.")

        # Donut composition insights
        elif chart_type == "Donut Chart" and x_col:
            counts = df[x_col].value_counts()
            if not counts.empty:
                top_cat, top_cnt = counts.index[0], counts.iloc[0]
                share_pct = round((top_cnt / len(df)) * 100, 1)
                insights.append(f"**Market/Share Leader:** '{top_cat}' accounts for **{share_pct}%** of total observations.")

    except Exception:
        pass

    return insights
