"""
DataPilot AI - AI Data Analyst Assistant
Translates natural-language business and statistical questions into grounded data calculations.
Uses Python/Pandas/DuckDB for exact mathematics (never hallucinated) and optionally 
enriches responses using Gemini, Groq, or OpenAI LLM APIs when keys are configured.
Provides concise answers with optional expandable calculation details.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_components import get_theme_colors, get_plotly_theme


def query_data_analyst(
    question: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    api_provider: str = "Google Gemini",
    theme_name: str = "Midnight Dark"
) -> Dict[str, Any]:
    """
    Core entrypoint for the AI Data Analyst.
    1. Interprets question intent.
    2. Performs exact mathematical computation on df.
    3. Produces a concise, plain-English summary starting with the key finding.
    4. Auto-generates accompanying visual Plotly chart where appropriate.
    5. Returns technical calculation steps for the 'How was this calculated?' section.
    """
    q = question.strip().lower()
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            "answer": "The dataset is empty. Please upload or select a valid dataset first.",
            "calculation_details": "No data available to calculate.",
            "facts": {},
            "table": None,
            "chart": None,
            "mode": "offline"
        }

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)

    # Detect matching columns in query text
    matched_nums = [c for c in num_cols if c.lower() in q or c.lower().replace("_", " ") in q]
    matched_cats = [c for c in cat_cols if c.lower() in q or c.lower().replace("_", " ") in q]

    target_num = matched_nums[0] if matched_nums else (num_cols[0] if num_cols else None)
    target_cat = matched_cats[0] if matched_cats else (cat_cols[0] if cat_cols else None)

    # -----------------------------------------------------------------------
    # INTENT 1: Correlation & Relationships
    # -----------------------------------------------------------------------
    if any(k in q for k in ["correlation", "correlate", "relationship", "correlated"]):
        if len(num_cols) < 2:
            return {
                "answer": f"Correlation requires at least 2 numerical columns, but this dataset only has: {num_cols}.",
                "calculation_details": "Need at least two numeric features to compute correlation.",
                "facts": {"numerical_columns": len(num_cols)},
                "table": None,
                "chart": None,
                "mode": "deterministic"
            }
        
        corr_matrix = df[num_cols].corr().round(3)
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((abs(val), val, c1, c2))
        
        pairs.sort(reverse=True)
        top_pair = pairs[0] if pairs else (0, 0, num_cols[0], num_cols[1])
        abs_r, r_val, c1, c2 = top_pair

        strength = "strong" if abs_r > 0.7 else ("moderate" if abs_r > 0.4 else "weak")
        direction = "positive" if r_val > 0 else "negative"

        facts = {
            "Strongest Pair": f"{c1} & {c2}",
            "Correlation (r)": r_val,
            "Relationship": f"{strength.capitalize()} {direction}"
        }
        
        template = get_plotly_theme(theme_name)
        colors = get_theme_colors(theme_name)
        chart = px.scatter(
            df, x=c1, y=c2,
            template=template,
            color_discrete_sequence=[colors[0]],
            title=f"Correlation: {c1} vs. {c2} (r = {r_val})"
        )
        chart.update_layout(height=360, margin=dict(l=35, r=35, t=40, b=35))

        concise_text = f"""**Key Finding:** **`{c1}`** and **`{c2}`** have the strongest relationship in this dataset (correlation score = **{r_val}**).

- As `{c1}` {'increases' if r_val > 0 else 'decreases'}, `{c2}` tends to {'increase' if r_val > 0 else 'decrease'}.
- Relationship strength: **{strength.capitalize()} {direction}**.
- *Note: High correlation shows a pattern, but does not mean one causes the other.*"""

        calc_details = f"Computed Pearson correlation coefficient matrix across all {len(num_cols)} numerical features. Highest off-diagonal pair: ({c1}, {c2}) with r = {r_val:.3f}."

        return _synthesize_response(question, concise_text, calc_details, facts, corr_matrix, chart, api_key, api_provider)

    # -----------------------------------------------------------------------
    # INTENT 2: Outliers & Anomalies
    # -----------------------------------------------------------------------
    if any(k in q for k in ["outlier", "anomaly", "unusual", "abnormal", "extreme"]):
        if not num_cols:
            return {
                "answer": "Cannot scan for outliers because the dataset contains no numerical columns.",
                "calculation_details": "Outlier detection requires numerical features.",
                "facts": {},
                "table": None,
                "chart": None,
                "mode": "deterministic"
            }

        target = target_num if target_num else num_cols[0]
        series = df[target].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_df = df[(df[target] < lower) | (df[target] > upper)]
        outlier_count = len(outliers_df)
        pct = round(outlier_count / len(df) * 100, 1)

        facts = {
            "Analyzed Feature": target,
            "Outlier Count": outlier_count,
            "Percentage": f"{pct}%",
            "Expected Range": f"[{lower:.1f} to {upper:.1f}]"
        }

        template = get_plotly_theme(theme_name)
        colors = get_theme_colors(theme_name)
        chart = px.box(
            df, y=target,
            points="outliers",
            template=template,
            color_discrete_sequence=[colors[0]],
            title=f"Outlier Inspection: {target}"
        )
        chart.update_layout(height=360, margin=dict(l=35, r=35, t=40, b=35))

        table_preview = outliers_df.head(8) if outlier_count > 0 else None

        concise_text = f"""**Key Finding:** Found **{outlier_count} unusual values** ({pct}% of rows) in **`{target}`** outside the expected range of **{lower:,.1f} to {upper:,.1f}**.

- Lowest value found: **{series.min():,.1f}**
- Highest value found: **{series.max():,.1f}**
- Review these records to decide whether they are valid extreme events or data errors."""

        calc_details = f"Calculated using Tukey's Interquartile Range (IQR) method with standard 1.5 multiplier. Q1 = {q1:.1f}, Q3 = {q3:.1f}, IQR = {iqr:.1f}. Lower fence: {lower:.1f}, Upper fence: {upper:.1f}."

        return _synthesize_response(question, concise_text, calc_details, facts, table_preview, chart, api_key, api_provider)

    # -----------------------------------------------------------------------
    # INTENT 3: Highest / Lowest / Top Contributing Categories
    # -----------------------------------------------------------------------
    if any(k in q for k in ["highest", "top", "most", "lowest", "bottom", "least", "contribute", "contribution", "share", "ranking"]):
        if target_cat and target_num:
            agg_type = "sum" if any(k in q for k in ["revenue", "sales", "total", "spend", "volume", "contribute"]) else "mean"
            grouped = df.groupby(target_cat)[target_num].agg(["count", agg_type]).reset_index()
            grouped.columns = [target_cat, "Records", f"{agg_type.capitalize()} of {target_num}"]
            
            is_asc = any(k in q for k in ["lowest", "bottom", "least"])
            grouped = grouped.sort_values(f"{agg_type.capitalize()} of {target_num}", ascending=is_asc).reset_index(drop=True)
            
            top_entity = grouped.iloc[0][target_cat]
            top_val = grouped.iloc[0][f"{agg_type.capitalize()} of {target_num}"]
            total_sum = df[target_num].sum()
            share_pct = round((top_val / total_sum * 100), 1) if (total_sum > 0 and agg_type == "sum") else 0

            facts = {
                "Category Feature": target_cat,
                "Metric": f"{agg_type.capitalize()} of {target_num}",
                "Leader": top_entity,
                "Value": round(top_val, 2)
            }

            template = get_plotly_theme(theme_name)
            colors = get_theme_colors(theme_name)
            chart = px.bar(
                grouped.head(8),
                x=target_cat,
                y=f"{agg_type.capitalize()} of {target_num}",
                text=f"{agg_type.capitalize()} of {target_num}",
                template=template,
                color_discrete_sequence=[colors[0]],
                title=f"Top {target_cat} by {agg_type.capitalize()} {target_num}"
            )
            chart.update_traces(texttemplate="%{text:.2s}", textposition="outside")
            chart.update_layout(height=360, margin=dict(l=35, r=35, t=40, b=35), xaxis=dict(tickangle=-25))

            concise_text = f"""**Key Finding:** **`{top_entity}`** ranks **#1** in **`{target_cat}`** with a total of **{top_val:,.1f}** {target_num.replace('_', ' ')}.

{f"- Accounts for **{share_pct}%** of all {target_num.replace('_', ' ')} across the entire dataset." if share_pct > 0 else ""}
- Total categories evaluated: **{len(grouped)}**."""

            calc_details = f"Grouped dataset by '{target_cat}' and calculated {agg_type} of '{target_num}'. Ordered results to identify leading category."

            return _synthesize_response(question, concise_text, calc_details, facts, grouped.head(8), chart, api_key, api_provider)

        elif target_cat:
            counts = df[target_cat].value_counts().reset_index()
            counts.columns = [target_cat, "Frequency"]
            top_entity = counts.iloc[0][target_cat]
            top_freq = counts.iloc[0]["Frequency"]
            pct = round(top_freq / total_rows * 100, 1)

            facts = {
                "Top Category": top_entity,
                "Frequency": top_freq,
                "Share": f"{pct}%"
            }

            template = get_plotly_theme(theme_name)
            colors = get_theme_colors(theme_name)
            chart = px.bar(
                counts.head(8), x=target_cat, y="Frequency", text="Frequency",
                template=template, color_discrete_sequence=[colors[1] if len(colors) > 1 else colors[0]],
                title=f"Frequency Breakdown for {target_cat}"
            )
            chart.update_traces(textposition="outside")
            chart.update_layout(height=360, margin=dict(l=35, r=35, t=40, b=35), xaxis=dict(tickangle=-25))

            concise_text = f"""**Key Finding:** **`{top_entity}`** is the most common category in **`{target_cat}`**, appearing **{top_freq:,} times** ({pct}% of all records)."""
            calc_details = f"Counted frequency of each unique category in '{target_cat}' across {total_rows:,} rows."

            return _synthesize_response(question, concise_text, calc_details, facts, counts.head(8), chart, api_key, api_provider)

    # -----------------------------------------------------------------------
    # INTENT 4: General Overview / Averages / Distributions
    # -----------------------------------------------------------------------
    if target_num:
        series = df[target_num].dropna()
        mean_v = series.mean()
        median_v = series.median()
        min_v = series.min()
        max_v = series.max()
        std_v = series.std()

        facts = {
            "Feature": target_num,
            "Mean": round(mean_v, 2),
            "Median": round(median_v, 2),
            "Min": round(min_v, 2),
            "Max": round(max_v, 2)
        }

        template = get_plotly_theme(theme_name)
        colors = get_theme_colors(theme_name)
        chart = px.histogram(
            df, x=target_num, nbins=25, marginal="box",
            template=template, color_discrete_sequence=[colors[0]],
            title=f"Distribution of {target_num}"
        )
        chart.update_layout(height=360, margin=dict(l=35, r=35, t=40, b=35))

        table_df = pd.DataFrame([facts]).T.reset_index()
        table_df.columns = ["Statistic", "Value"]

        concise_text = f"""**Key Finding:** The average **`{target_num}`** is **{mean_v:,.1f}**, with values ranging from **{min_v:,.1f}** to **{max_v:,.1f}**.

- Midpoint (Median): **{median_v:,.1f}**
- Spread (Std Dev): **{std_v:,.1f}**
- {'The mean is higher than the median, indicating a few high values pull the average up.' if mean_v > median_v * 1.1 else ('The mean is lower than the median, indicating low values pull the average down.' if mean_v < median_v * 0.9 else 'The values are evenly balanced around the center.')}"""

        calc_details = f"Calculated mean, median, standard deviation, and min/max boundaries across {len(series):,} valid rows in '{target_num}'."

        return _synthesize_response(question, concise_text, calc_details, facts, table_df, chart, api_key, api_provider)

    # Fallback response
    return {
        "answer": f"Found {total_rows:,} records and {len(df.columns)} columns ({', '.join(df.columns[:6])}). Please name a specific column to examine.",
        "calculation_details": "Inspected dataset schema and column list.",
        "facts": {"total_rows": total_rows, "columns_count": len(df.columns)},
        "table": df.head(5),
        "chart": None,
        "mode": "deterministic"
    }


def _synthesize_response(
    question: str,
    concise_text: str,
    calc_details: str,
    facts: Dict[str, Any],
    table: Optional[pd.DataFrame],
    chart: Optional[go.Figure],
    api_key: Optional[str],
    api_provider: str
) -> Dict[str, Any]:
    """If API key is present, calls LLM for a short natural summary. Otherwise returns concise text."""
    if not api_key:
        return {
            "answer": concise_text,
            "calculation_details": calc_details,
            "facts": facts,
            "table": table,
            "chart": chart,
            "mode": "Deterministic Calculations"
        }

    try:
        if "gemini" in api_provider.lower():
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""You are DataPilot AI. Answer the user question concisely using ONLY these facts:
Question: "{question}"
Facts: {facts}
Calculations: {concise_text}

Instructions:
1. Start with the most important finding in bold.
2. Provide 2 short bullet points with exact numbers.
3. Keep response under 70 words. Do not use jargon."""

            response = model.generate_content(prompt)
            return {
                "answer": response.text.strip(),
                "calculation_details": calc_details,
                "facts": facts,
                "table": table,
                "chart": chart,
                "mode": f"AI-Enriched ({api_provider})"
            }

        elif "groq" in api_provider.lower():
            from groq import Groq
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are DataPilot AI. Provide short, concise answers starting with the main finding in bold. Max 70 words."},
                    {"role": "user", "content": f"Question: {question}\nFacts: {facts}\nSummary: {concise_text}"}
                ],
                temperature=0.2,
                max_tokens=150
            )
            return {
                "answer": completion.choices[0].message.content,
                "calculation_details": calc_details,
                "facts": facts,
                "table": table,
                "chart": chart,
                "mode": f"AI-Enriched ({api_provider})"
            }

    except Exception as e:
        return {
            "answer": concise_text,
            "calculation_details": f"{calc_details} (Note: LLM unavailable, used local calculation engine: {str(e)})",
            "facts": facts,
            "table": table,
            "chart": chart,
            "mode": "Deterministic Calculations"
        }

    return {
        "answer": concise_text,
        "calculation_details": calc_details,
        "facts": facts,
        "table": table,
        "chart": chart,
        "mode": "Deterministic Calculations"
    }
