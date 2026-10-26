"""
DataPilot AI - DuckDB Analytical SQL Workspace & Security Sandbox
Provides high-performance analytical SQL execution, strict security validation,
rule-based & AI-assisted Natural Language-to-SQL generation, and beginner query explanations.
"""

import time
import re
from typing import Dict, List, Tuple, Any, Optional
import duckdb
import pandas as pd


# Dangerous SQL keywords and filesystem operations blocked in analytical sandbox
DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "ATTACH", "DETACH", "INSTALL", "LOAD", "PRAGMA", "CALL",
    "COPY", "CHECKPOINT", "IMPORT", "EXPORT", "READ_CSV", "WRITE_CSV",
    "READ_PARQUET", "WRITE_PARQUET", "READ_JSON", "WRITE_JSON"
]


class SQLExecutionEngine:
    """
    Manages DuckDB in-memory database instance, table registration,
    and query execution profiling.
    """
    def __init__(self):
        # Create dedicated in-memory DuckDB connection
        self.conn = duckdb.connect(database=":memory:")

    def register_table(self, table_name: str, df: pd.DataFrame):
        """Registers a pandas DataFrame as an analytical in-memory DuckDB table."""
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        self.conn.register(clean_name, df)
        # Also alias to 'data' for convenient default queries
        self.conn.register("data", df)
        self.conn.register("uploaded_table", df)

    def execute_query(self, query: str, max_rows: int = 1000) -> Tuple[Optional[pd.DataFrame], float, Optional[str]]:
        """
        Executes a validated read-only SQL query against the registered tables.
        Returns (result_df, execution_time_ms, error_message).
        """
        is_safe, security_msg = validate_sql_security(query)
        if not is_safe:
            return None, 0.0, f"Security Violation: {security_msg}"

        start_time = time.perf_counter()
        try:
            # Execute query and fetch as Arrow/DataFrame
            result_df = self.conn.execute(query).fetchdf()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            # Apply safety limit on returned rows to prevent UI crashing on million-row queries
            if len(result_df) > max_rows:
                result_df = result_df.head(max_rows)
                
            return result_df, elapsed_ms, None
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            # Format beginner-friendly syntax error
            error_str = str(e)
            return None, elapsed_ms, error_str


def validate_sql_security(query: str) -> Tuple[bool, Optional[str]]:
    """
    Validates that a SQL query is strictly read-only and analytical.
    Blocks arbitrary filesystem access, schema modifications, and non-SELECT statements.
    """
    cleaned = query.strip()
    # Strip comments
    cleaned = re.sub(r"--.*?\n", " ", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
    cleaned_upper = cleaned.upper().strip()

    if not cleaned_upper:
        return False, "Query is empty."

    # Must begin with SELECT, WITH, or EXPLAIN
    if not (cleaned_upper.startswith("SELECT") or cleaned_upper.startswith("WITH") or cleaned_upper.startswith("EXPLAIN")):
        return False, "Only read-only analytical statements ('SELECT' or 'WITH') are permitted in the workspace."

    # Check for prohibited dangerous keywords
    for kw in DANGEROUS_KEYWORDS:
        # Use regex boundary check so that a column named 'update_date' is not blocked
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, cleaned_upper):
            return False, f"The keyword or function '{kw}' is blocked for security to prevent filesystem or state modifications."

    # Prevent multiple chained queries with semicolons
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple queries separated by semicolons are not permitted. Please run one query at a time."

    return True, None


def explain_sql_query(query: str) -> Dict[str, str]:
    """
    Breaks down a SQL query into logical clauses and explains what each clause 
    does in beginner-friendly academic language.
    """
    explanation = {}
    q_clean = query.strip()
    
    # Check SELECT
    select_match = re.search(r"SELECT\s+(.*?)\s+FROM", q_clean, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        columns = select_match.group(1).strip()
        explanation["SELECT Clause"] = f"Specifies which attributes or aggregations to retrieve ({columns})."

    # Check FROM
    from_match = re.search(r"FROM\s+([a-zA-Z0-9_]+)", q_clean, flags=re.IGNORECASE)
    if from_match:
        table = from_match.group(1).strip()
        explanation["FROM Clause"] = f"Designates the source analytical table '{table}' as the underlying dataset."

    # Check WHERE
    where_match = re.search(r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)", q_clean, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        cond = where_match.group(1).strip()
        explanation["WHERE Clause"] = f"Filters the raw records before aggregation: only rows satisfying ({cond}) are included."

    # Check GROUP BY
    group_match = re.search(r"GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER\s+BY|\s+LIMIT|$)", q_clean, flags=re.IGNORECASE | re.DOTALL)
    if group_match:
        dim = group_match.group(1).strip()
        explanation["GROUP BY Clause"] = f"Groups identical values in ({dim}) into single summary rows for aggregation."

    # Check HAVING
    having_match = re.search(r"HAVING\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|$)", q_clean, flags=re.IGNORECASE | re.DOTALL)
    if having_match:
        h_cond = having_match.group(1).strip()
        explanation["HAVING Clause"] = f"Filters aggregated group results ({h_cond}) after the GROUP BY computation."

    # Check ORDER BY
    order_match = re.search(r"ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)", q_clean, flags=re.IGNORECASE | re.DOTALL)
    if order_match:
        sort_col = order_match.group(1).strip()
        desc = "descending order (highest first)" if "DESC" in sort_col.upper() else "ascending order (lowest first)"
        explanation["ORDER BY Clause"] = f"Sorts the final analytical output by {sort_col} in {desc}."

    # Check LIMIT
    limit_match = re.search(r"LIMIT\s+(\d+)", q_clean, flags=re.IGNORECASE)
    if limit_match:
        n = limit_match.group(1).strip()
        explanation["LIMIT Clause"] = f"Restricts the output to the top {n} records for performance."

    if not explanation:
        explanation["General"] = "Standard analytical SQL statement querying in-memory DuckDB table."

    return explanation


def generate_suggested_queries(df: pd.DataFrame, table_name: str = "data") -> List[Dict[str, str]]:
    """
    Auto-generates relevant analytical SQL queries tailored to the current dataset's column names.
    """
    suggestions = []
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    
    # 1. Dataset Overview
    suggestions.append({
        "title": "Dataset Shape & Record Count",
        "sql": f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT {cat_cols[0] if cat_cols else '*'}) AS unique_entities FROM {table_name};",
        "description": "Calculates total rows and unique counts."
    })

    # 2. Aggregation by Category
    if len(cat_cols) > 0 and len(num_cols) > 0:
        c = cat_cols[0]
        n = num_cols[0]
        suggestions.append({
            "title": f"Summary Statistics for {n} by {c}",
            "sql": f"""SELECT 
    {c},
    COUNT(*) AS record_count,
    ROUND(AVG({n}), 2) AS avg_{n},
    ROUND(SUM({n}), 2) AS total_{n},
    ROUND(MIN({n}), 2) AS min_{n},
    ROUND(MAX({n}), 2) AS max_{n}
FROM {table_name}
GROUP BY {c}
ORDER BY total_{n} DESC
LIMIT 10;""",
            "description": f"Calculates count, mean, sum, min, and max of '{n}' grouped by '{c}'."
        })

    # 3. Top 5 Records
    if len(num_cols) > 0:
        n = num_cols[0]
        display_cols = (cat_cols[:2] + [n]) if cat_cols else [n]
        suggestions.append({
            "title": f"Top 5 Records by {n}",
            "sql": f"SELECT {', '.join(display_cols)} FROM {table_name} ORDER BY {n} DESC LIMIT 5;",
            "description": f"Retrieves highest ranking observations sorted by {n}."
        })

    # 4. Missing Value Scan
    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if missing_cols:
        clauses = [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS missing_{c}" for c in missing_cols[:4]]
        suggestions.append({
            "title": "Missing Value Audit Query",
            "sql": f"SELECT\n    {',\n    '.join(clauses)}\nFROM {table_name};",
            "description": "Computes null counts across features using conditional SQL aggregation."
        })

    # 5. Window Function / Running Total
    if len(num_cols) > 1 and len(cat_cols) > 0:
        n1 = num_cols[0]
        c = cat_cols[0]
        suggestions.append({
            "title": "Analytical Window Ranking (DENSE_RANK)",
            "sql": f"""SELECT 
    {c},
    {n1},
    DENSE_RANK() OVER (ORDER BY {n1} DESC) AS ranking_position
FROM {table_name}
WHERE {n1} IS NOT NULL
LIMIT 10;""",
            "description": "Demonstrates SQL analytical window function for competitive ranking."
        })

    return suggestions


def rule_based_nl_to_sql(question: str, df: pd.DataFrame, table_name: str = "data") -> Optional[str]:
    """
    Translates common natural language analytical questions into valid DuckDB SQL
    using deterministic heuristic parsing.
    """
    q = question.lower().strip()
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)

    # Find referenced columns in question
    referenced_nums = [c for c in num_cols if c.lower() in q or c.lower().replace("_", " ") in q]
    referenced_cats = [c for c in cat_cols if c.lower() in q or c.lower().replace("_", " ") in q]

    target_num = referenced_nums[0] if referenced_nums else (num_cols[0] if num_cols else None)
    target_cat = referenced_cats[0] if referenced_cats else (cat_cols[0] if cat_cols else None)

    # Pattern: Top N by Column
    top_match = re.search(r"top\s+(\d+)", q)
    limit_n = int(top_match.group(1)) if top_match else 5

    if any(k in q for k in ["highest", "top", "greatest", "maximum", "best"]) and target_num:
        if target_cat and any(k in q for k in ["by", "per", "in each"]):
            return f"""SELECT {target_cat}, ROUND(SUM({target_num}), 2) AS total_{target_num}
FROM {table_name}
GROUP BY {target_cat}
ORDER BY total_{target_num} DESC
LIMIT {limit_n};"""
        elif target_cat:
            return f"SELECT {target_cat}, {target_num} FROM {table_name} ORDER BY {target_num} DESC LIMIT {limit_n};"
        else:
            return f"SELECT * FROM {table_name} ORDER BY {target_num} DESC LIMIT {limit_n};"

    # Pattern: Lowest / Bottom
    if any(k in q for k in ["lowest", "bottom", "least", "minimum", "worst"]) and target_num:
        if target_cat and any(k in q for k in ["by", "per", "in each"]):
            return f"""SELECT {target_cat}, ROUND(AVG({target_num}), 2) AS avg_{target_num}
FROM {table_name}
GROUP BY {target_cat}
ORDER BY avg_{target_num} ASC
LIMIT {limit_n};"""
        else:
            return f"SELECT * FROM {table_name} ORDER BY {target_num} ASC LIMIT {limit_n};"

    # Pattern: Average / Mean
    if any(k in q for k in ["average", "mean", "avg"]) and target_num:
        if target_cat:
            return f"""SELECT {target_cat}, ROUND(AVG({target_num}), 2) AS avg_{target_num}
FROM {table_name}
GROUP BY {target_cat}
ORDER BY avg_{target_num} DESC;"""
        else:
            return f"SELECT ROUND(AVG({target_num}), 2) AS avg_{target_num} FROM {table_name};"

    # Pattern: Total / Sum
    if any(k in q for k in ["total", "sum", "revenue", "overall"]) and target_num:
        if target_cat:
            return f"""SELECT {target_cat}, ROUND(SUM({target_num}), 2) AS total_{target_num}
FROM {table_name}
GROUP BY {target_cat}
ORDER BY total_{target_num} DESC;"""
        else:
            return f"SELECT ROUND(SUM({target_num}), 2) AS total_{target_num} FROM {table_name};"

    # Pattern: Count / How many
    if any(k in q for k in ["how many", "count", "frequency", "distribution"]):
        if target_cat:
            return f"""SELECT {target_cat}, COUNT(*) AS count
FROM {table_name}
GROUP BY {target_cat}
ORDER BY count DESC
LIMIT 15;"""
        else:
            return f"SELECT COUNT(*) AS total_records FROM {table_name};"

    # Default fallback
    if target_cat and target_num:
        return f"""SELECT {target_cat}, ROUND(AVG({target_num}), 2) AS avg_{target_num}, COUNT(*) AS total
FROM {table_name}
GROUP BY {target_cat}
ORDER BY avg_{target_num} DESC
LIMIT 10;"""

    return f"SELECT * FROM {table_name} LIMIT 10;"
