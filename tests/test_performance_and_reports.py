"""
Performance benchmark and full dossier export tests
"""

import time
import pytest
import pandas as pd
from modules.data_loader import load_sample_dataset, get_dataset_metadata
from modules.sql_engine import SQLExecutionEngine
from modules.insights_engine import generate_comprehensive_insights
from modules.report_generator import generate_pdf_report, generate_excel_dossier
from modules.cleaning_engine import CleaningSession, apply_remove_duplicates

def test_50k_benchmark_performance():
    start = time.perf_counter()
    df, err = load_sample_dataset("performance_benchmark")
    load_time = time.perf_counter() - start
    
    assert err is None
    assert len(df) == 50000
    print(f"Loaded 50,000 rows in {load_time:.2f}s")

    # DuckDB SQL aggregation performance on 50,000 rows
    engine = SQLExecutionEngine()
    engine.register_table("benchmark", df)
    
    sql_query = """
    SELECT 
        Region, 
        Product_Line, 
        COUNT(*) AS tx_count, 
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(AVG(Revenue), 2) AS avg_order_val
    FROM benchmark
    GROUP BY Region, Product_Line
    ORDER BY total_revenue DESC;
    """
    res_df, runtime_ms, err = engine.execute_query(sql_query)
    assert err is None
    assert len(res_df) == 30  # 5 regions * 6 products = 30 groups
    print(f"Executed DuckDB aggregation over 50,000 rows in {runtime_ms} ms")
    assert runtime_ms < 500  # Sub-second performance

def test_pdf_and_excel_export_generation():
    df, _ = load_sample_dataset("ecommerce_churn")
    session = CleaningSession(df)
    apply_remove_duplicates(session)
    
    meta = get_dataset_metadata(session.current_df)
    insights = generate_comprehensive_insights(session.current_df)
    
    # 1. Generate PDF
    pdf_bytes = generate_pdf_report(
        metadata=meta,
        audit_trail=session.audit_trail,
        insights=insights,
        ml_results=[{"Model": "Random Forest", "Accuracy": 0.88, "F1-Score": 0.87}]
    )
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    print(f"Generated PDF report: {len(pdf_bytes)} bytes")

    # 2. Generate Excel Dossier
    excel_bytes = generate_excel_dossier(
        cleaned_df=session.current_df,
        audit_trail=session.audit_trail,
        ml_results=[{"Model": "Random Forest", "Accuracy": 0.88, "F1-Score": 0.87}]
    )
    assert len(excel_bytes) > 2000
    print(f"Generated multi-sheet Excel dossier: {len(excel_bytes)} bytes")
