"""
DataPilot AI - Automated Insights & Data Storytelling Engine
Generates statistically grounded insights, identifies category dominance,
strong correlations, anomalies, and provides responsible business takeaways.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def generate_comprehensive_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans the dataset and synthesizes deep, responsible analytical insights.
    Every finding includes the mathematical metric, formula/method, and explicit risk caveats.
    """
    insights = []
    total_rows = len(df)
    if total_rows == 0:
        return insights

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)

    # 1. Dataset Scale & Completeness
    total_cells = total_rows * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    completeness_pct = round((1 - (missing_cells / total_cells)) * 100, 1) if total_cells > 0 else 100.0

    insights.append({
        "type": "Health & Scale",
        "badge": "Dataset Integrity",
        "title": "Scale and Information Density",
        "finding": f"The dataset encapsulates {total_rows:,} records across {len(df.columns)} features with an overall data completeness of {completeness_pct}%.",
        "supporting_metric": f"{completeness_pct}% Completeness",
        "methodology": "Total non-null cells divided by matrix dimensions (Rows × Columns)",
        "takeaway": "High data completeness minimizes imputation bias and strengthens the validity of downstream statistical inferences.",
        "icon": "🌐"
    })

    # 2. Category Concentration (Pareto Principle / Dominance)
    if cat_cols:
        for cat in cat_cols[:2]:
            series = df[cat].dropna().astype(str)
            if series.nunique() > 1:
                top_item = series.value_counts().index[0]
                top_cnt = series.value_counts().iloc[0]
                share = round((top_cnt / total_rows) * 100, 1)

                if share >= 30.0:
                    insights.append({
                        "type": "Category Dominance",
                        "badge": "High Concentration",
                        "title": f"Dominant Segment in '{cat}': {top_item}",
                        "finding": f"'{top_item}' represents {top_cnt:,} records, commanding {share}% of the entire '{cat}' feature.",
                        "supporting_metric": f"{share}% Market/Category Share",
                        "methodology": "Relative frequency calculation: count(category) / total_valid_records",
                        "takeaway": f"Business performance and risk are heavily concentrated in '{top_item}'. Consider segment-specific operational strategies.",
                        "icon": "🏆"
                    })

    # 3. Notable Correlations (with mandatory non-causality disclosure)
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        found_corr = False
        
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                r = corr_matrix.loc[c1, c2]
                if abs(r) >= 0.5:
                    found_corr = True
                    direction = "positive" if r > 0 else "negative"
                    strength = "strong" if abs(r) >= 0.7 else "moderate"
                    
                    insights.append({
                        "type": "Statistical Correlation",
                        "badge": "Pairwise Association",
                        "title": f"{strength.capitalize()} {direction.capitalize()} Correlation: {c1} & {c2}",
                        "finding": f"Pearson correlation coefficient r = {r:.3f} indicates that as '{c1}' increases, '{c2}' tends to {'increase' if r > 0 else 'decrease'}.",
                        "supporting_metric": f"r = {r:.3f}",
                        "methodology": "Pearson Product-Moment Correlation (covariance normalized by standard deviations)",
                        "takeaway": "Correlation does not imply causation. Do not assume changing one metric will automatically cause changes in the other without experimental verification.",
                        "icon": "🔗"
                    })
                    break
            if found_corr:
                break

    # 4. Skewness and Distribution Symmetry
    if num_cols:
        for num in num_cols[:2]:
            series = df[num].dropna()
            if len(series) > 5:
                skew = series.skew()
                mean_v = series.mean()
                median_v = series.median()
                
                if abs(skew) > 1.0:
                    direction = "Right-skewed (positive tail)" if skew > 0 else "Left-skewed (negative tail)"
                    insights.append({
                        "type": "Distribution Skewness",
                        "badge": "Non-Normal Distribution",
                        "title": f"High Asymmetry Detected in '{num}'",
                        "finding": f"'{num}' exhibits a skewness score of {skew:.2f}. The mean ({mean_v:,.2f}) differs noticeably from the median ({median_v:,.2f}).",
                        "supporting_metric": f"Skewness = {skew:.2f}",
                        "methodology": "Fisher-Pearson coefficient of skewness (3rd standardized moment)",
                        "takeaway": f"Because this distribution is {direction}, the median is a more reliable measure of central tendency than the arithmetic mean.",
                        "icon": "📐"
                    })

    # 5. Outlier Concentrations
    if num_cols:
        for num in num_cols:
            series = df[num].dropna()
            if len(series) > 10:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
                    out_cnt = len(outliers)
                    out_pct = round(out_cnt / len(series) * 100, 1)
                    
                    if out_pct >= 3.0:
                        insights.append({
                            "type": "Anomaly Alert",
                            "badge": "Outlier Concentration",
                            "title": f"Extreme Values in '{num}'",
                            "finding": f"{out_cnt} records ({out_pct}% of column) fall outside Tukey's IQR boundaries [{q1 - 1.5*iqr:.1f}, {q3 + 1.5*iqr:.1f}].",
                            "supporting_metric": f"{out_cnt} Outliers ({out_pct}%)",
                            "methodology": "Tukey's Fences: Values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR",
                            "takeaway": "Examine whether these represent VIP clients / high-volume events or measurement errors. Winsorizing or robust scaling is advised before linear modeling.",
                            "icon": "⚠️"
                        })
                        break

    # 6. Strategic Recommendations
    insights.append({
        "type": "Strategic Recommendations",
        "badge": "Analytical Action Items",
        "title": "Recommended Next Analytical Stages",
        "finding": "Dataset is prepared for segmentation, cohort analysis, and supervised machine learning.",
        "supporting_metric": "3 Action Items",
        "methodology": "Heuristic Data Science Workflow Protocol",
        "takeaway": "1. Resolve identified missing values via the Smart Cleaning page.\n2. Leverage the SQL workspace to aggregate metrics by core dimensions.\n3. Train baseline predictive models in the ML Lab to benchmark feature importance.",
        "icon": "🎯"
    })

    return insights
