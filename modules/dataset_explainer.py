"""
DataPilot AI - Intelligent Dataset Understanding & Semantic Profiler
Automatically analyzes dataset structure, file metadata, and column attributes
to generate plain-English explanations, column dictionaries, and tailored analysis suggestions.
Supports user context injection, local heuristic inference, and optional privacy-preserving LLM synthesis.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


# Dictionary of known semantic patterns for column meanings
COLUMN_SEMANTIC_RULES = [
    # Identifiers
    (r"^(id|.*_id|.*_code|.*_key|.*_num|code|hash|uuid|ref|reference|tracking.*)$", "Unique identifier or reference code"),
    # Demographics
    (r"^(age|years_old)$", "Age of the subject in years"),
    (r"^(gender|sex)$", "Gender or biological sex classification"),
    (r"^(name|full_name|first_name|last_name|emp_name|cust_name|patient_name)$", "Personal name or entity label"),
    (r"^(city|state|country|region|location|address|zip|postal.*)$", "Geographical location or operational region"),
    (r"^(education|education_level|degree)$", "Highest level of formal education completed"),
    # HR / Workforce
    (r"^(salary|monthly_salary|annual_salary|wage|compensation|pay)$", "Compensation or salary earnings amount"),
    (r"^(department|dept|division|team|business_unit)$", "Organizational department or business division"),
    (r"^(role|job_role|job|title|position|designation)$", "Occupational role or job title"),
    (r"^(experience|work_experience|years_experience|experience_years|years_at_company|tenure|service_years)$", "Total work experience or company tenure in years"),
    (r"^(joining_date|join_date|hire_date|start_date|date_of_joining)$", "Date of employment commencement"),
    (r"^(performance_rating|perf_score|rating|evaluation|score)$", "Performance evaluation rating score"),
    (r"^(overtime|ot)$", "Indicator of overtime work status (e.g., Yes/No)"),
    (r"^(attrition|left_company|terminated|resigned)$", "Employee departure or retention status (potential target)"),
    # Sales & E-Commerce
    (r"^(revenue|total_revenue|sales|total_sales|spend|spending|total_spend)$", "Monetary revenue or spending amount"),
    (r"^(unit_price|price|cost|amount)$", "Unit selling price or item cost"),
    (r"^(units_sold|quantity|qty|units|total_purchases|purchases)$", "Quantity of units or number of purchases"),
    (r"^(discount|discount_used|promo.*)$", "Discount or promotional offer applied"),
    (r"^(product|product_line|item|preferred_category|category)$", "Merchandise category or product classification"),
    (r"^(payment_method|payment|payment_type)$", "Payment method or transaction channel"),
    (r"^(churn|churned|is_churn)$", "Customer churn indicator (potential target)"),
    (r"^(store|store_id|branch)$", "Retail store or branch identifier"),
    # Healthcare / Clinical
    (r"^(bmi|body_mass.*)$", "Body Mass Index (ratio of weight to height squared)"),
    (r"^(blood_pressure|bp|systolic|diastolic)$", "Blood pressure measurement in mmHg"),
    (r"^(glucose|glucose_level|blood_sugar)$", "Blood glucose concentration level"),
    (r"^(cholesterol|chol)$", "Serum cholesterol level in mg/dL"),
    (r"^(smoking|smoking_status|smoker)$", "Tobacco consumption or smoking habit classification"),
    (r"^(physical_activity.*|exercise.*|activity_hours)$", "Weekly physical activity or exercise hours"),
    (r"^(family_history.*)$", "Indicator of family medical history predisposition"),
    (r"^(health_risk.*|risk_score)$", "Calculated clinical health risk assessment score"),
    (r"^(diabetic_status|diabetes|diagnosis|disease.*)$", "Clinical diagnosis or disease classification status"),
    # Temporal
    (r".*(date|time|timestamp|created_at|updated_at|year|month).*", "Calendar date or temporal event timestamp")
]


def _infer_semantic_meaning(col_name: str, series: pd.Series) -> Tuple[str, bool]:
    """
    Infers likely real-world meaning of a column from its name and observed characteristics.
    Returns (meaning_string, is_confident).
    If meaning is unclear, explicitly returns 'Meaning could not be confidently inferred'.
    """
    clean_name = col_name.strip().lower()
    norm_name = re.sub(r"[\s\-]+", "_", clean_name)

    for pattern, description in COLUMN_SEMANTIC_RULES:
        if re.search(pattern, clean_name) or re.search(pattern, norm_name):
            return description, True

    # Check for binary flags
    if series.nunique() == 2 and series.dropna().isin([0, 1, "0", "1", "Yes", "No", "True", "False"]).all():
        return f"Binary indicator flag for '{col_name}'", True

    # Fallback when pattern is ambiguous
    return "Meaning could not be confidently inferred", False


def _sanitize_example_value(series: pd.Series, col_name: str) -> str:
    """
    Selects a representative sample value while masking potentially sensitive personal identifiers.
    """
    valid = series.dropna()
    if valid.empty:
        return "None (all values missing)"
    
    val = valid.iloc[0]
    c_lower = col_name.lower()

    # Mask personal names or sensitive IDs if necessary
    if any(k in c_lower for k in ["ssn", "password", "secret", "token"]):
        return "••••••••"
    
    val_str = str(val).strip()
    if len(val_str) > 35:
        val_str = val_str[:32] + "..."
    return val_str


def _detect_dataset_domain(columns: List[str], file_name: str = "") -> Tuple[str, str, str]:
    """
    Infers the thematic domain of the dataset (e.g. HR, Healthcare, Retail/Sales, Finance).
    Returns (domain_name, entity_phrase, domain_summary_phrase).
    """
    cols_joined = " ".join(columns).lower() + " " + file_name.lower()

    # Domain indicators
    is_health = any(k in cols_joined for k in ["patient", "medical", "health", "glucose", "bmi", "cholesterol", "diabetic", "blood_pressure", "clinical"])
    is_hr = any(k in cols_joined for k in ["employee", "salary", "department", "job_role", "attrition", "tenure", "hr", "workforce", "overtime", "joining_date", "experience"])
    is_ecommerce = any(k in cols_joined for k in ["customer", "churn", "purchase", "product", "store", "discount", "transaction", "sales", "order", "retail"])
    is_finance = any(k in cols_joined for k in ["revenue", "budget", "profit", "expense", "stock", "portfolio", "banking", "account", "loan"])

    if is_health:
        return "Healthcare & Clinical", "patient health records", "medical health records and clinical diagnostic indicators"
    elif is_hr:
        return "Human Resources & Workforce", "employee information", "employee human resources records, compensation, and workforce metrics"
    elif is_ecommerce:
        return "E-Commerce & Retail", "customer transactions and commercial records", "customer transactions, purchase behavior, and commercial activity"
    elif is_finance:
        return "Finance & Accounting", "financial accounts and transactions", "financial accounts, monetary transactions, and fiscal metrics"
    else:
        return "General Analytical", "analytical tabular records", "structured tabular records with numerical and categorical features"


def _build_dynamic_exploration_goals(columns: List[str]) -> List[str]:
    """
    Derives 2-3 specific, dynamic exploration goals directly from the available column features.
    """
    cols_norm = [re.sub(r"[\s\-]+", "_", c.lower()) for c in columns]
    cols_joined = " ".join(cols_norm)
    goals = []

    # Goal 1: Distribution across segments/departments/categories
    if any(k in cols_joined for k in ["department", "dept", "division"]):
        goals.append("workforce distribution")
    elif any(k in cols_joined for k in ["category", "product_line", "product"]):
        goals.append("product category distribution")
    elif any(k in cols_joined for k in ["city", "region", "country", "state"]):
        goals.append("geographic segment distribution")

    # Goal 2: Financial/Monetary/Compensation patterns
    if any(k in cols_joined for k in ["salary", "wage", "compensation", "pay"]):
        goals.append("salary patterns")
    elif any(k in cols_joined for k in ["revenue", "sales", "spend", "spending", "price", "amount"]):
        goals.append("spending and revenue patterns")

    # Goal 3: Experience, Tenure, or Temporal trends
    if any(k in cols_joined for k in ["experience", "tenure", "years_at_company", "service_years"]):
        goals.append("employee experience")
    elif any(k in cols_joined for k in ["date", "joining_date", "year", "month"]):
        goals.append("historical trends over time")

    # Goal 4: Retention or Diagnostic risk
    if any(k in cols_joined for k in ["attrition", "churn"]):
        goals.append("retention and churn drivers")
    elif any(k in cols_joined for k in ["health_risk", "risk", "glucose", "blood_pressure", "bmi"]):
        goals.append("clinical health risk factors")
    elif any(k in cols_joined for k in ["performance_rating", "rating", "perf_score"]):
        goals.append("performance benchmarks")

    # Fallback if no domain-specific targets matched
    if len(goals) < 2:
        if any(c for c in columns if "date" in c.lower() or "year" in c.lower()):
            goals.append("chronological patterns")
        goals.append("categorical distributions")
        goals.append("feature relationships")

    return goals[:3]


def generate_dataset_overview(
    df: pd.DataFrame,
    file_name: str = "Uploaded Dataset",
    sheet_name: Optional[str] = None,
    user_context: str = "",
    api_key: Optional[str] = None,
    api_provider: str = "Google Gemini"
) -> Dict[str, Any]:
    """
    Primary engine for Intelligent Dataset Understanding.
    Analyzes columns, types, sample values, and generates:
    - What this data is about (concise, plain-English explanation)
    - Column dictionary with detected type and inferred meaning
    - Actionable analysis suggestions tailored to available features
    - Support for optional user-provided context
    """
    total_rows, total_cols = len(df), len(df.columns)
    col_names = list(df.columns)

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime"]).columns)
    bool_cols = list(df.select_dtypes(include=["bool"]).columns)

    # Detect identifier columns
    id_cols = [c for c in col_names if re.search(r"(id|code|key|uuid|ref)", c.lower()) or df[c].nunique() == total_rows]

    # Detect domain
    domain_title, entity_phrase, domain_phrase = _detect_dataset_domain(col_names, file_name)

    # Build Column Dictionary
    column_dictionary = []
    for col in col_names:
        series = df[col]
        dtype_str = str(series.dtype)
        if pd.api.types.is_datetime64_any_dtype(series):
            detected_type = "Date / Time"
        elif pd.api.types.is_integer_dtype(series):
            detected_type = "Integer"
        elif pd.api.types.is_float_dtype(series):
            detected_type = "Decimal (Float)"
        elif pd.api.types.is_bool_dtype(series):
            detected_type = "Boolean (True/False)"
        else:
            detected_type = "Text / Categorical"

        meaning, is_confident = _infer_semantic_meaning(col, series)
        example_val = _sanitize_example_value(series, col)
        null_cnt = int(series.isnull().sum())
        null_pct = round((null_cnt / total_rows) * 100, 1) if total_rows > 0 else 0

        column_dictionary.append({
            "name": col,
            "detected_type": detected_type,
            "semantic_meaning": meaning,
            "is_ai_inferred": is_confident,
            "example_value": example_val,
            "missing_count": null_cnt,
            "missing_pct": null_pct
        })

    # Generate Analysis Suggestions
    suggestions = []
    
    # Suggestion 1: Categorical Comparison
    if cat_cols and num_cols:
        c_target = cat_cols[0]
        n_target = num_cols[0]
        suggestions.append({
            "title": f"Compare '{n_target}' across '{c_target}' segments",
            "description": f"Group data by '{c_target}' to examine average and total '{n_target}' variations.",
            "icon": "📊"
        })

    # Suggestion 2: Temporal Trends
    if date_cols and num_cols:
        d_target = date_cols[0]
        n_target = num_cols[0]
        suggestions.append({
            "title": f"Examine changes over time ({d_target})",
            "description": f"Analyze chronological trajectory, seasonal peaks, and growth patterns in '{n_target}'.",
            "icon": "📈"
        })
    elif any("date" in c.lower() or "year" in c.lower() for c in col_names) and num_cols:
        d_cand = [c for c in col_names if "date" in c.lower() or "year" in c.lower()][0]
        suggestions.append({
            "title": f"Parse '{d_cand}' for time-series analysis",
            "description": f"Convert '{d_cand}' to datetime format to unlock trend analysis and chronological sorting.",
            "icon": "📅"
        })

    # Suggestion 3: Numerical Relationships
    if len(num_cols) >= 2:
        suggestions.append({
            "title": f"Evaluate correlation between '{num_cols[0]}' and '{num_cols[1]}'",
            "description": "Investigate whether these two numerical variables co-vary positively or negatively.",
            "icon": "🔗"
        })

    # Suggestion 4: Data Quality & Completeness
    missing_total = int(df.isnull().sum().sum())
    dup_total = int(df.duplicated().sum())
    if missing_total > 0 or dup_total > 0:
        suggestions.append({
            "title": "Review data quality in Smart Cleaning",
            "description": f"Resolve {missing_total:,} missing values and {dup_total:,} duplicate rows to ensure clean downstream modeling.",
            "icon": "🧹"
        })

    # Suggestion 5: Predictive Target Modeling
    outcome_candidates = [c for c in col_names if re.search(r"(attrition|churn|risk|status|target|label|default)", c.lower())]
    numeric_targets = [c for c in col_names if re.search(r"(price|revenue|salary|cost|amount|spend)", c.lower())]
    target_candidates = outcome_candidates + numeric_targets

    if target_candidates:
        t_cand = target_candidates[0]
        is_classification = df[t_cand].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[t_cand])
        task_str = "Classification" if is_classification else "Regression"
        suggestions.append({
            "title": f"Train predictive models for '{t_cand}' ({task_str})",
            "description": f"Use eligible features in the Machine Learning Lab to benchmark baseline predictive models.",
            "icon": "🧪"
        })

    # Build Plain-English Dataset Description
    important_features = [c for c in col_names if c not in id_cols][:5]
    if not important_features:
        important_features = col_names[:4]

    if len(important_features) > 1:
        features_str = ", ".join(important_features[:-1]) + f", and {important_features[-1]}"
    else:
        features_str = important_features[0]

    exploration_goals = _build_dynamic_exploration_goals(col_names)
    if len(exploration_goals) > 1:
        goals_str = ", ".join(exploration_goals[:-1]) + f", and {exploration_goals[-1]}"
    else:
        goals_str = exploration_goals[0]

    if user_context.strip():
        inferred_description = (
            f"Based on your context ('{user_context.strip()}'), this dataset appears to contain {entity_phrase}, "
            f"including {features_str}. It can be used to explore {goals_str}."
        )
    else:
        inferred_description = (
            f"This dataset appears to contain {entity_phrase}, "
            f"including {features_str}. It can be used to explore {goals_str}."
        )

    # Optional minimal privacy-preserving LLM enrichment
    if api_key:
        try:
            enriched_desc = _call_llm_for_overview(
                file_name=file_name,
                columns=col_names[:25],
                user_context=user_context,
                api_key=api_key,
                api_provider=api_provider
            )
            if enriched_desc:
                inferred_description = enriched_desc
        except Exception:
            # Fallback gracefully to heuristic description
            pass

    return {
        "description": inferred_description,
        "domain": domain_title,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "types_summary": {
            "numerical": len(num_cols),
            "categorical": len(cat_cols),
            "datetime": len(date_cols),
            "boolean": len(bool_cols),
            "identifiers": len(id_cols)
        },
        "column_dictionary": column_dictionary,
        "suggestions": suggestions,
        "user_context": user_context
    }


def _call_llm_for_overview(
    file_name: str,
    columns: List[str],
    user_context: str,
    api_key: str,
    api_provider: str
) -> Optional[str]:
    """
    Sends minimal metadata (column names and file name ONLY, zero raw data rows)
    to generate an articulate, plain-English dataset description.
    """
    prompt = f"""You are DataPilot AI. Summarize what this dataset appears to represent in 2 plain-English sentences:
File Name: {file_name}
Column Names: {columns}
User Context (if any): {user_context}

Guidelines:
- Start with 'This dataset appears to contain...'
- Mention 3-4 key features and what analyses can be explored.
- Do not make absolute claims; state what the data appears to represent.
- Maximum 50 words."""

    if "gemini" in api_provider.lower():
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    elif "groq" in api_provider.lower():
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        return completion.choices[0].message.content.strip()
    return None
