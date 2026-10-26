"""

DataPilot AI — Your All-in-One Data Analyst Workspace
Academic M.Sc. Data Science Project
Main Application Entrypoint & Multi-Page Router
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="DataPilot AI — Data Analyst Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from modules.ui_components import (
    load_custom_css, apply_theme_css, render_page_header, render_workflow_stepper,
    render_kpi_card, get_theme_colors, get_plotly_theme, THEMES,
    render_datapilot_brand_logo, render_datapilot_topbar, render_html,
    render_flux_brand_logo, render_flux_topbar, render_flux_sidebar_promo,
    render_flux_bubble_card, render_flux_kpi_card,
    render_flux_wellness_matrix_card, render_flux_dark_card
)
from modules.data_loader import (
    load_dataset, load_sample_dataset, get_dataset_metadata,
    get_paginated_preview, get_excel_sheet_names, AVAILABLE_SAMPLES
)
from modules.cleaning_engine import (
    CleaningSession, detect_cleaning_issues, apply_remove_duplicates,
    apply_missing_imputation, apply_drop_missing, apply_text_standardization,
    apply_date_parsing, apply_outlier_handling, apply_drop_column,
    get_cleaning_comparison_stats
)
from modules.profiler import (
    get_numerical_summary, get_categorical_summary,
    plot_correlation_heatmap, plot_missing_values, plot_column_distribution
)
from modules.sql_engine import (
    SQLExecutionEngine, validate_sql_security, explain_sql_query,
    generate_suggested_queries, rule_based_nl_to_sql
)
from modules.ai_analyst import query_data_analyst
from modules.insights_engine import generate_comprehensive_insights
from modules.ml_lab import (
    auto_detect_ml_task, check_target_leakage, MLTrainingPipeline, get_available_models
)
from modules.report_generator import generate_pdf_report, generate_excel_dossier
from modules.dataset_explainer import generate_dataset_overview
from modules.chart_guide import (
    ANALYTICAL_GOALS, CHART_GOALS,
    inspect_dataset_for_visualization,
    get_chart_recommendations, recommend_charts_for_goal,
    generate_guided_chart, derive_chart_insights
)


# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

def init_session_state():
    """Initializes persistent workspace session state variables."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Enterprise Dark"
    if "cleaning_session" not in st.session_state:
        st.session_state.cleaning_session = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = "No dataset loaded"
    if "sql_engine" not in st.session_state:
        st.session_state.sql_engine = SQLExecutionEngine()
    if "ml_pipeline" not in st.session_state:
        st.session_state.ml_pipeline = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
    if "api_provider" not in st.session_state:
        st.session_state.api_provider = "Google Gemini"
    if "current_page" not in st.session_state or st.session_state.current_page == "🏠 Home":
        st.session_state.current_page = "🏠 Home / Overview"
    if "dismissed_issues" not in st.session_state:
        st.session_state.dismissed_issues = set()
    if "dataset_overview" not in st.session_state:
        st.session_state.dataset_overview = None
    if "user_dataset_context" not in st.session_state:
        st.session_state.user_dataset_context = ""
    if "custom_dataset_description" not in st.session_state:
        st.session_state.custom_dataset_description = None


init_session_state()

# Apply theme CSS dynamically
load_custom_css()


# ---------------------------------------------------------------------------
# HELPER DATA ACCESSORS
# ---------------------------------------------------------------------------

def get_current_df() -> pd.DataFrame:
    """Returns the current cleaned DataFrame or None."""
    if st.session_state.cleaning_session is not None:
        return st.session_state.cleaning_session.current_df
    return None


def get_raw_df() -> pd.DataFrame:
    """Returns the original uncleaned DataFrame or None."""
    if st.session_state.cleaning_session is not None:
        return st.session_state.cleaning_session.raw_df
    return None


def set_new_dataset(df: pd.DataFrame, name: str, sheet_name=None):
    """Initializes a new dataset session and registers tables."""
    session = CleaningSession(df)
    st.session_state.cleaning_session = session
    st.session_state.dataset_name = name
    st.session_state.sql_engine.register_table("data", df)
    st.session_state.ml_pipeline = None
    st.session_state.dismissed_issues = set()
    st.session_state.custom_dataset_description = None
    st.session_state.dataset_overview = generate_dataset_overview(
        df=df,
        file_name=name,
        sheet_name=sheet_name,
        user_context=st.session_state.get("user_dataset_context", ""),
        api_key=st.session_state.get("api_key", ""),
        api_provider=st.session_state.get("api_provider", "Google Gemini")
    )


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION (REFERENCE-MATCHED CHARCOAL #202124)
# ---------------------------------------------------------------------------

with st.sidebar:
    render_datapilot_brand_logo()

    # Active dataset status pill
    curr_df = get_current_df()
    if curr_df is not None:
        meta = get_dataset_metadata(curr_df)
        render_html(f"""
        <div style="background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.28); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-size: 11.5px; font-weight: 600; color: #60A5FA; text-transform: uppercase; letter-spacing: 0.5px;">Active Dataset</div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {st.session_state.dataset_name[:24]}
            </div>
            <div style="font-size: 12.5px; color: #94A3B8; margin-top: 2px;">{len(curr_df):,} rows</div>
        </div>
        """)
    else:
        render_html("""
        <div style="background: #181B24; border: 1px solid #2E3440; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
            <div style="font-size: 13.5px; font-weight: 500; color: #FFFFFF;">No dataset loaded</div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">Upload a file to begin</div>
        </div>
        """)

    pages = [
        "🏠 Home / Overview",
        "📤 Upload Dataset",
        "🔍 Data Explorer",
        "🧹 Smart Data Cleaning",
        "📊 Chart Guide",
        "⚡ SQL Assistant",
        "🤖 AI Data Analyst",
        "🧪 Machine Learning Lab",
        "📑 Reports & Exports",
        "⚙️ Settings & Theme"
    ]
    
    # Handle backward-compatible page mapping
    curr_pg = st.session_state.current_page
    if curr_pg == "🏠 Home":
        curr_pg = "🏠 Home / Overview"
    elif curr_pg == "📤 Upload & Understand":
        curr_pg = "📤 Upload Dataset"

    selected_page = st.radio(
        "Workspace Navigation",
        pages,
        index=pages.index(curr_pg) if curr_pg in pages else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_page = selected_page

    st.divider()
    theme_choice = st.selectbox(
        "Theme Palette",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme) if st.session_state.theme in THEMES else 0,
        key="sidebar_theme_picker"
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    render_html("""
    <div style="padding: 8px 4px 4px 4px; text-align: center;">
        <span style="font-size: 12px; color: #94A3B8; font-weight: 500;">
            DataPilot AI v2.4 • M.Sc. Data Science
        </span>
    </div>
    """)


# ---------------------------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# ---------------------------------------------------------------------------

if st.session_state.current_page in ["🏠 Home / Overview", "🏠 Home"]:
    curr_df = get_current_df()
    meta = get_dataset_metadata(curr_df) if curr_df is not None else None

    # Sleek Enterprise Dark Top Bar Header
    render_datapilot_topbar(
        title="Welcome to DataPilot AI",
        subtitle="Understand, clean, explore, and analyze your data in one unified workspace.",
        dataset_name=st.session_state.dataset_name if curr_df is not None else None,
        row_count=meta['total_rows'] if meta else None,
        health_score=meta['health_score'] if meta else None
    )

    # Subtle Linear-style Workflow Stepper
    render_workflow_stepper("Home" if curr_df is not None else "Upload")

    # Balanced 2-Column Grid matching Linear/Vercel SaaS proportions
    home_c1, home_c2 = st.columns([1.2, 1])

    with home_c1:
        # Unified Ingestion & Benchmark Dataset Card
        with st.container(border=True):
            render_html("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Upload Dataset</h2>
                <span class="badge-blue">CSV &amp; Excel</span>
            </div>
            <p style="font-size: 14px; color: #94A3B8; margin: 0 0 16px 0; line-height: 1.5;">
                Select a spreadsheet to inspect distributions, detect quality flaws, and run in-memory queries.
            </p>
            """)

            home_uploader = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="home_file_uploader",
                label_visibility="collapsed",
                help="Supports CSV files and Excel workbooks (.xlsx, .xls)."
            )

            if home_uploader is not None:
                f_ext = home_uploader.name.split(".")[-1].lower()
                s_name = None
                if f_ext in ["xlsx", "xls"]:
                    sheets = get_excel_sheet_names(home_uploader)
                    if len(sheets) > 1:
                        s_name = st.selectbox("Select Excel Sheet:", sheets, key="home_sheet_select")
                    elif sheets:
                        s_name = sheets[0]

                if st.button("Load and Profile Dataset", type="primary", key="btn_home_ingest", use_container_width=True):
                    with st.spinner(f"Ingesting '{home_uploader.name}'..."):
                        df, err = load_dataset(home_uploader, file_type=f_ext, sheet_name=s_name)
                        if df is not None:
                            d_name = f"{home_uploader.name} ({s_name})" if s_name else home_uploader.name
                            set_new_dataset(df, d_name, sheet_name=s_name)
                            st.session_state.current_page = "📤 Upload Dataset"
                            st.rerun()
                        else:
                            st.error(f"Error loading file: {err}")

            render_html("""
            <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid #2E3440;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 13px; font-weight: 600; color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.5px;">Quick Benchmark Datasets</span>
                    <span class="badge-neutral">Instant Sample</span>
                </div>
            </div>
            """)

            sample_c1, sample_c2 = st.columns(2)
            with sample_c1:
                if st.button("Employee HR (Excel)", use_container_width=True):
                    with st.spinner("Loading Employee HR dataset..."):
                        df, err = load_sample_dataset("employee_attrition", sheet_name="Employees")
                        if df is not None:
                            set_new_dataset(df, "Employee HR & Attrition", sheet_name="Employees")
                            st.rerun()
                        else:
                            st.error(err)
                
                if st.button("E-Commerce Churn (CSV)", use_container_width=True):
                    with st.spinner("Loading E-Commerce dataset..."):
                        df, err = load_sample_dataset("ecommerce_churn")
                        if df is not None:
                            set_new_dataset(df, "E-Commerce Customer Churn")
                            st.rerun()
                        else:
                            st.error(err)

            with sample_c2:
                if st.button("Retail Benchmark (50K)", use_container_width=True):
                    with st.spinner("Loading 50K benchmark dataset..."):
                        df, err = load_sample_dataset("performance_benchmark")
                        if df is not None:
                            set_new_dataset(df, "Retail Performance Benchmark (50K)")
                            st.rerun()
                        else:
                            st.error(err)

                if st.button("Patient Health Risk (CSV)", use_container_width=True):
                    with st.spinner("Loading Medical dataset..."):
                        df, err = load_sample_dataset("patient_health")
                        if df is not None:
                            set_new_dataset(df, "Medical Patient Health Risk")
                            st.rerun()
                        else:
                            st.error(err)

    with home_c2:
        if curr_df is None:
            # Workspace Workflow Summary Card matching height & padding
            with st.container(border=True):
                render_html("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Workspace Workflow</h2>
                    <span class="badge-blue">4 Core Stages</span>
                </div>
                <p style="font-size: 14px; color: #94A3B8; margin: 0 0 16px 0; line-height: 1.5;">
                    DataPilot AI guides you through four core analytical stages:
                </p>

                <div style="display: flex; flex-direction: column; gap: 13px; margin-bottom: 20px;">
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <div style="width: 24px; height: 24px; border-radius: 6px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 1px;">1</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Ingest Data</div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Upload CSV/Excel files or load pre-packaged benchmark datasets.</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <div style="width: 24px; height: 24px; border-radius: 6px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 1px;">2</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Explore &amp; Profile</div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Review statistical summaries, distributions, and correlation matrices.</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <div style="width: 24px; height: 24px; border-radius: 6px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 1px;">3</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Clean &amp; Repair</div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Detect quality flaws and apply audited transformations.</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <div style="width: 24px; height: 24px; border-radius: 6px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 1px;">4</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Analyze &amp; Predict</div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Execute SQL queries, guided visualizations, or train ML models.</div>
                        </div>
                    </div>
                </div>

                <div style="border-top: 1px solid #2E3440; padding-top: 14px; display: flex; gap: 8px;">
                    <span class="badge-blue">In-Memory Engine</span>
                    <span class="badge-neutral">DuckDB Sandboxed</span>
                </div>
                """)

        else:
            # Active Dataset Overview Panel
            clean_rows = meta['total_rows'] - meta['duplicate_rows']
            clean_pct = round((clean_rows / max(1, meta['total_rows'])) * 100, 1)

            with st.container(border=True):
                render_html(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Active Dataset</h2>
                    <span class="badge-blue">{meta['health_score']}% Quality</span>
                </div>
                <div style="font-size: 15px; font-weight: 600; color: #FFFFFF; margin-bottom: 14px;">
                    {st.session_state.dataset_name}
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;">
                    <div class="content-card-secondary">
                        <div style="font-size: 12.5px; font-weight: 500; color: #94A3B8;">Total Rows</div>
                        <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['total_rows']:,}</div>
                    </div>
                    <div class="content-card-secondary">
                        <div style="font-size: 12.5px; font-weight: 500; color: #94A3B8;">Columns</div>
                        <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['total_cols']}</div>
                    </div>
                    <div class="content-card-secondary">
                        <div style="font-size: 12.5px; font-weight: 500; color: #94A3B8;">Health Score</div>
                        <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['health_score']}/100</div>
                    </div>
                    <div class="content-card-secondary">
                        <div style="font-size: 12.5px; font-weight: 500; color: #94A3B8;">Memory Usage</div>
                        <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['memory_str']}</div>
                    </div>
                </div>

                <div style="border-top: 1px solid #2E3440; padding-top: 12px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: #94A3B8; margin-bottom: 6px;">
                        <span>Data Integrity</span>
                        <span style="font-weight: 600; color: #FFFFFF;">{clean_pct}% clean</span>
                    </div>
                    <div style="width: 100%; height: 6px; background: #2E3440; border-radius: 9999px; overflow: hidden;">
                        <div style="width: {clean_pct}%; height: 100%; background: #3B82F6; border-radius: 9999px;"></div>
                    </div>
                </div>
                """)

                # Contextual Actions
                st.markdown("<p style='font-size: 12.5px; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px 0;'>Next Steps</p>", unsafe_allow_html=True)
                if st.button("Go to Data Explorer →", type="primary", use_container_width=True):
                    st.session_state.current_page = "🔍 Data Explorer"
                    st.rerun()

                a_c1, a_c2 = st.columns(2)
                with a_c1:
                    if st.button("Smart Cleaning", use_container_width=True):
                        st.session_state.current_page = "🧹 Smart Data Cleaning"
                        st.rerun()
                    if st.button("SQL Assistant", use_container_width=True):
                        st.session_state.current_page = "⚡ SQL Assistant"
                        st.rerun()
                with a_c2:
                    if st.button("Chart Guide", use_container_width=True):
                        st.session_state.current_page = "📊 Chart Guide"
                        st.rerun()
                    if st.button("Machine Learning", use_container_width=True):
                        st.session_state.current_page = "🧪 Machine Learning Lab"
                        st.rerun()

# ---------------------------------------------------------------------------
# PAGE 2: UPLOAD DATASET
# ---------------------------------------------------------------------------

elif st.session_state.current_page in ["📤 Upload & Understand", "📤 Upload Dataset"]:
    render_page_header(
        title="Upload Your Dataset",
        subtitle="Upload a CSV or Excel spreadsheet to profile schema types, detect data-quality flaws, and begin your analysis.",
        badge="Ingestion Stage"
    )

    curr_df = get_current_df()

    # Balanced 2-Column Composition matching Linear / Vercel Layout
    up_c1, up_c2 = st.columns([1.2, 1])

    with up_c1:
        with st.container(border=True):
            render_html("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Select Data File</h2>
                <span class="badge-blue">CSV &amp; Excel</span>
            </div>
            <p style="font-size: 14px; color: #94A3B8; margin: 0 0 16px 0; line-height: 1.5;">
                Drag and drop your spreadsheet file below to inspect columns, detect quality issues, and start querying.
            </p>
            """)

            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="main_file_uploader",
                label_visibility="collapsed",
                help="Supports standard comma/tab-separated CSVs and Excel workbooks (.xlsx, .xls)."
            )

            render_html("""
            <div style="margin-top: 12px; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: #94A3B8;">
                <span>📁 <strong style="color: #FFFFFF;">Formats:</strong> .csv, .xlsx, .xls, .tsv</span>
                <span>⚡ <strong style="color: #FFFFFF;">Max File Size:</strong> 200MB</span>
                <span>🔒 <strong style="color: #FFFFFF;">Processing:</strong> Local DuckDB session</span>
            </div>
            """)

            if uploaded_file is not None:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                sheet_name = None

                if file_ext in ["xlsx", "xls"]:
                    available_sheets = get_excel_sheet_names(uploaded_file)
                    if len(available_sheets) > 1:
                        sheet_name = st.selectbox("Select Excel Sheet to Load:", available_sheets)
                    elif available_sheets:
                        sheet_name = available_sheets[0]

                if st.button("📥 Process and Ingest Dataset", type="primary", use_container_width=True):
                    with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                        df, err = load_dataset(uploaded_file, file_type=file_ext, sheet_name=sheet_name)
                        if df is not None:
                            display_name = f"{uploaded_file.name} ({sheet_name})" if sheet_name else uploaded_file.name
                            set_new_dataset(df, display_name, sheet_name=sheet_name)
                            st.success(f"Successfully loaded '{display_name}' ({len(df):,} rows, {len(df.columns)} columns)!")
                            st.rerun()
                        else:
                            st.error(f"Error loading file: {err}")

    with up_c2:
        if curr_df is None:
            # Clean Quick Start Sample Datasets Panel
            with st.container(border=True):
                render_html("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Quick Start Datasets</h2>
                    <span class="badge-neutral">Instant Load</span>
                </div>
                <p style="font-size: 14px; color: #94A3B8; margin: 0 0 12px 0; line-height: 1.5;">
                    Don't have a dataset ready? Load a curated academic benchmark with realistic data quality flaws:
                </p>
                """)

                for key, s_meta in AVAILABLE_SAMPLES.items():
                    s_col1, s_col2 = st.columns([3.8, 1.2])
                    with s_col1:
                        st.markdown(f"<div style='padding-top: 3px;'><strong style='font-size: 14px; color: #FFFFFF;'>{s_meta['name']}</strong><div style='font-size: 12.5px; color: #94A3B8;'>{s_meta['badge']} • {s_meta['description'][:52]}...</div></div>", unsafe_allow_html=True)
                    with s_col2:
                        if st.button("Load", key=f"btn_quick_{key}", use_container_width=True):
                            with st.spinner(f"Loading {s_meta['name']}..."):
                                sheet = "Employees" if key == "employee_attrition" else None
                                df, err = load_sample_dataset(key, sheet_name=sheet)
                                if df is not None:
                                    set_new_dataset(df, s_meta["name"], sheet_name=sheet)
                                    st.success(f"Loaded '{s_meta['name']}'!")
                                    st.rerun()
                                else:
                                    st.error(err)
                    st.markdown("<div style='height: 1px; background: #2E3440; margin: 4px 0 6px 0;'></div>", unsafe_allow_html=True)
        else:
            # Active Dataset Status Card
            meta = get_dataset_metadata(curr_df)
            clean_rows = meta['total_rows'] - meta['duplicate_rows']
            clean_pct = round((clean_rows / max(1, meta['total_rows'])) * 100, 1)

            with st.container(border=True):
                render_html(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Currently Loaded</h2>
                    <span class="badge-blue">{meta['health_score']}% Quality</span>
                </div>
                <div style="font-size: 15px; font-weight: 600; color: #FFFFFF; margin-bottom: 12px; word-break: break-word;">
                    {st.session_state.dataset_name}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
                    <div class="content-card-secondary" style="margin-bottom: 0; padding: 10px 12px;">
                        <div style="font-size: 12px; color: #94A3B8; font-weight: 500;">Rows</div>
                        <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['total_rows']:,}</div>
                    </div>
                    <div class="content-card-secondary" style="margin-bottom: 0; padding: 10px 12px;">
                        <div style="font-size: 12px; color: #94A3B8; font-weight: 500;">Columns</div>
                        <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{meta['total_cols']}</div>
                    </div>
                </div>
                <div style="border-top: 1px solid #2E3440; padding-top: 10px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: #94A3B8; margin-bottom: 4px;">
                        <span>Data Integrity</span>
                        <span style="font-weight: 600; color: #FFFFFF;">{clean_pct}% clean</span>
                    </div>
                    <div style="width: 100%; height: 6px; background: #2E3440; border-radius: 9999px; overflow: hidden;">
                        <div style="width: {clean_pct}%; height: 100%; background: #3B82F6; border-radius: 9999px;"></div>
                    </div>
                </div>
                """)

                if st.button("Explore Dataset Now →", type="primary", key="btn_jump_explorer", use_container_width=True):
                    st.session_state.current_page = "🔍 Data Explorer"
                    st.rerun()

                c_nav1, c_nav2 = st.columns(2)
                with c_nav1:
                    if st.button("Clean Data", key="btn_jump_clean", use_container_width=True):
                        st.session_state.current_page = "🧹 Smart Data Cleaning"
                        st.rerun()
                with c_nav2:
                    if st.button("View Charts", key="btn_jump_charts", use_container_width=True):
                        st.session_state.current_page = "📊 Chart Guide"
                        st.rerun()

    # When a dataset IS loaded: Display Readable Preview and Relevant Information
    if curr_df is not None:
        # Ensure dataset overview is computed
        if st.session_state.get("dataset_overview") is None:
            st.session_state.dataset_overview = generate_dataset_overview(
                df=curr_df,
                file_name=st.session_state.dataset_name,
                user_context=st.session_state.get("user_dataset_context", ""),
                api_key=st.session_state.get("api_key", ""),
                api_provider=st.session_state.get("api_provider", "Google Gemini")
            )
        
        overview = st.session_state.dataset_overview
        display_desc = st.session_state.custom_dataset_description or overview.get("description", "")

        # -------------------------------------------------------------------
        # 1. READABLE INTERACTIVE DATASET PREVIEW (PAGINATED)
        # -------------------------------------------------------------------
        with st.container(border=True):
            page_size = 15
            total_pages = max(1, (len(curr_df) + page_size - 1) // page_size)

            p_h1, p_h2 = st.columns([7, 5])
            with p_h1:
                render_html("""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 18px;">📄</span>
                    <h2 style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 0;">Interactive Dataset Preview</h2>
                </div>
                """)
            with p_h2:
                prev_c1, prev_c2 = st.columns([1, 2])
                with prev_c1:
                    page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1, key="upload_preview_page", label_visibility="collapsed")
                with prev_c2:
                    st.markdown(f"<div style='font-size: 13px; color: #94A3B8; padding-top: 6px;'>Showing rows {(page_num-1)*page_size + 1}–{min(page_num*page_size, len(curr_df))} of {len(curr_df):,}</div>", unsafe_allow_html=True)

            preview_slice, _ = get_paginated_preview(curr_df, page=page_num, page_size=page_size)
            st.dataframe(preview_slice, use_container_width=True)

        # -------------------------------------------------------------------
        # 2. ABOUT THIS DATASET (PLAIN-ENGLISH INFERRED EXPLANATION)
        # -------------------------------------------------------------------
        render_html(f"""
        <div class="content-card" style="margin-top: 14px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 18px;">💡</span>
                    <span style="font-size: 18px; font-weight: 600; color: #FFFFFF;">About This Dataset</span>
                </div>
                <span class="badge-blue">Inferred Semantic Overview</span>
            </div>
            <div style="font-size: 14.5px; line-height: 1.6; color: #E2E8F0; background: #181B24; border: 1px solid #2E3440; border-radius: 8px; padding: 14px 16px;">
                "{display_desc}"
            </div>
        </div>
        """)

        # User correction and context customization
        with st.expander("✏️ Correct Inferred Description or Provide Project Context", expanded=False):
            st.caption("If the system misunderstands your dataset or you want to provide custom background (e.g. 'This dataset contains monthly sales records for my college project'), update below:")
            u_ctx = st.text_input(
                "Optional Project Context:",
                value=st.session_state.user_dataset_context,
                placeholder="e.g., Monthly sales records for my college data science project.",
                key="input_user_context"
            )
            u_desc = st.text_area(
                "Dataset Description (editable):",
                value=display_desc,
                height=90,
                key="input_custom_desc"
            )
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            with btn_col1:
                if st.button("💾 Save Description", use_container_width=True):
                    st.session_state.custom_dataset_description = u_desc.strip()
                    st.session_state.user_dataset_context = u_ctx.strip()
                    st.success("Description saved successfully!")
                    st.rerun()
            with btn_col2:
                if st.button("🔄 Re-Infer with Context", use_container_width=True):
                    st.session_state.user_dataset_context = u_ctx.strip()
                    st.session_state.dataset_overview = generate_dataset_overview(
                        df=curr_df,
                        file_name=st.session_state.dataset_name,
                        user_context=u_ctx.strip(),
                        api_key=st.session_state.get("api_key", ""),
                        api_provider=st.session_state.get("api_provider", "Google Gemini")
                    )
                    st.session_state.custom_dataset_description = None
                    st.success("Re-inferred dataset understanding with updated context!")
                    st.rerun()
            with btn_col3:
                if st.button("↺ Reset to Original", use_container_width=True):
                    st.session_state.custom_dataset_description = None
                    st.session_state.user_dataset_context = ""
                    st.session_state.dataset_overview = generate_dataset_overview(
                        df=curr_df,
                        file_name=st.session_state.dataset_name
                    )
                    st.rerun()

        # -------------------------------------------------------------------
        # 3. CONCISE DATASET OVERVIEW METRICS
        # -------------------------------------------------------------------
        st.markdown("### 📊 Dataset Overview")
        types_sum = overview.get("types_summary", {})
        
        ov_col1, ov_col2, ov_col3, ov_col4, ov_col5 = st.columns(5)
        with ov_col1:
            render_kpi_card("Dataset Size", f"{overview.get('total_rows', len(curr_df)):,} rows", delta=f"{overview.get('total_cols', len(curr_df.columns))} cols", icon="📏")
        with ov_col2:
            render_kpi_card("Inferred Domain", overview.get("domain", "General Analytical"), icon="🏢")
        with ov_col3:
            render_kpi_card("Numerical", f"{types_sum.get('numerical', 0)} cols", icon="📈")
        with ov_col4:
            render_kpi_card("Categorical", f"{types_sum.get('categorical', 0)} cols", icon="🏷️")
        with ov_col5:
            d_cnt = types_sum.get('datetime', 0)
            id_cnt = types_sum.get('identifiers', 0)
            render_kpi_card("Dates & IDs", f"{d_cnt} dates", delta=f"{id_cnt} IDs", icon="📅")

        # -------------------------------------------------------------------
        # 4. WHAT CAN BE ANALYZED (DYNAMIC TAILORED SUGGESTIONS)
        # -------------------------------------------------------------------
        suggestions = overview.get("suggestions", [])
        if suggestions:
            st.markdown("### 🎯 What Can Be Analyzed")
            sug_cols = st.columns(min(len(suggestions), 3))
            for idx, sug in enumerate(suggestions):
                with sug_cols[idx % len(sug_cols)]:
                    render_html(f"""
                    <div style="background: #1E222D; border: 1px solid #2E3440; border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; height: 100%;">
                        <div style="font-size: 15px; font-weight: 600; color: #FFFFFF; margin-bottom: 6px;">
                            {sug.get('icon', '💡')} {sug['title']}
                        </div>
                        <div style="font-size: 14px; color: #94A3B8; line-height: 1.5;">
                            {sug['description']}
                        </div>
                    </div>
                    """)

        # -------------------------------------------------------------------
        # 5. COLUMN DICTIONARY (EXPLAIN COLUMNS IN SIMPLE LANGUAGE)
        # -------------------------------------------------------------------
        st.markdown("### 📖 Column Dictionary (What Each Column Means)")
        st.caption("Distinguishes between **Detected Data Type** (programmatic type from data inspection) and **Likely Meaning** (semantic interpretation inferred from naming patterns and sample distributions).")
        
        col_dict = overview.get("column_dictionary", [])
        if col_dict:
            dict_df = pd.DataFrame(col_dict)
            dict_df_display = pd.DataFrame({
                "Column Name": dict_df["name"],
                "Detected Data Type": dict_df["detected_type"],
                "Likely Meaning": dict_df["semantic_meaning"],
                "Sample Value": dict_df["example_value"],
                "Missing Values": dict_df.apply(lambda r: f"{r['missing_count']:,} ({r['missing_pct']}%)", axis=1)
            })
            st.dataframe(
                dict_df_display,
                use_container_width=True,
                hide_index=True
            )


# ---------------------------------------------------------------------------
# PAGE 3: DATA EXPLORER
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "🔍 Data Explorer":
    render_page_header(
        title="Exploratory Data Analysis & Profiling",
        subtitle="Uncover distribution shapes, statistical quartiles, correlation coefficients, and categorical cardinality.",
        badge="Profiling Stage"
    )
    render_workflow_stepper("Profile")

    curr_df = get_current_df()
    if curr_df is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        tab_dict, tab_num, tab_cat, tab_corr, tab_miss, tab_single = st.tabs([
            "📖 Column Dictionary & Overview",
            "📈 Numerical Statistics",
            "🏷️ Categorical Profiles",
            "🔗 Correlation Heatmap",
            "❓ Missingness Inspection",
            "🔎 Single Feature Deep-Dive"
        ])

        with tab_dict:
            st.markdown("#### Intelligent Dataset Understanding & Column Dictionary:")
            if st.session_state.get("dataset_overview") is None:
                st.session_state.dataset_overview = generate_dataset_overview(
                    df=curr_df,
                    file_name=st.session_state.dataset_name,
                    user_context=st.session_state.get("user_dataset_context", ""),
                    api_key=st.session_state.get("api_key", ""),
                    api_provider=st.session_state.get("api_provider", "Google Gemini")
                )
            overview = st.session_state.dataset_overview
            display_desc = st.session_state.custom_dataset_description or overview.get("description", "")
            
            st.info(f"💡 **About this dataset:** {display_desc}")

            col_dict = overview.get("column_dictionary", [])
            if col_dict:
                dict_df = pd.DataFrame(col_dict)
                dict_df_display = pd.DataFrame({
                    "Column Name": dict_df["name"],
                    "Detected Data Type": dict_df["detected_type"],
                    "Likely Meaning": dict_df["semantic_meaning"],
                    "Sample Value": dict_df["example_value"],
                    "Missing Values": dict_df.apply(lambda r: f"{r['missing_count']:,} ({r['missing_pct']}%)", axis=1)
                })
                st.dataframe(dict_df_display, use_container_width=True, hide_index=True)
            
            st.caption("Detected Data Type is programmatically determined; Likely Meaning is an inferred suggestion.")

        with tab_num:
            st.markdown("#### Numerical Descriptive Statistics:")
            num_summary = get_numerical_summary(curr_df)
            if not num_summary.empty:
                st.dataframe(num_summary, use_container_width=True)
                st.caption("Includes skewness, zero counts, and IQR outlier counts.")
            else:
                st.info("No numerical columns found in this dataset.")

        with tab_cat:
            st.markdown("#### Categorical & Cardinality Profiles:")
            cat_summary = get_categorical_summary(curr_df)
            if not cat_summary.empty:
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical or text columns found in this dataset.")

        with tab_corr:
            st.markdown("#### Feature Correlation Matrix:")
            method = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True)
            corr_fig = plot_correlation_heatmap(curr_df, method=method, theme_name=st.session_state.theme)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
                st.caption("Note: Correlation evaluates co-variation; it does not establish causal dependency.")
            else:
                st.info("At least 2 numerical features are required to compute a correlation matrix.")

        with tab_miss:
            st.markdown("#### Missing Values by Feature:")
            miss_fig = plot_missing_values(curr_df, theme_name=st.session_state.theme)
            if miss_fig:
                st.plotly_chart(miss_fig, use_container_width=True)
            else:
                st.success("🎉 Excellent! This dataset currently contains zero missing values.")

        with tab_single:
            st.markdown("#### Single Feature Distribution Deep-Dive:")
            chosen_col = st.selectbox("Select Column to Analyze:", curr_df.columns)
            dist_fig = plot_column_distribution(curr_df, chosen_col, theme_name=st.session_state.theme)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 4: SMART DATA CLEANING — SIMPLIFIED USER EXPERIENCE
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "🧹 Smart Data Cleaning":
    render_page_header(
        title="Smart Data Cleaning",
        subtitle="Detect data quality problems, review clear recommendations, and apply user-controlled fixes with concise explanations.",
        badge="Cleaning Stage"
    )
    render_workflow_stepper("Clean")

    curr_df = get_current_df()
    raw_df = get_raw_df()
    session = st.session_state.cleaning_session

    if curr_df is None or session is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        diff_stats = get_cleaning_comparison_stats(raw_df, curr_df)

        # -------------------------------------------------------------------
        # 1. SHOW A SHORT SUMMARY FIRST (REQUIREMENT 3)
        # -------------------------------------------------------------------
        if session.audit_trail:
            st.markdown("### ✅ Cleaning completed")
            for bullet in session.get_compact_summary():
                st.markdown(f"- {bullet}")
        else:
            detected = detect_cleaning_issues(curr_df)
            if not detected:
                st.success("Your data looks good. No cleaning changes were necessary.")
            else:
                st.info("Quality issues detected. Review recommendations below to clean your dataset.")

        # Simple before-and-after comparison with important counts
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card(
                "Total Rows",
                f"{diff_stats['curr_rows']:,}",
                delta=f"{diff_stats['rows_diff']}" if diff_stats['rows_diff'] != 0 else None,
                delta_text="from original",
                icon="🔢"
            )
        with c2:
            render_kpi_card(
                "Missing Values",
                f"{diff_stats['curr_missing']:,}",
                delta=f"{diff_stats['missing_diff']}" if diff_stats['missing_diff'] != 0 else None,
                delta_text="from original",
                icon="❓"
            )
        with c3:
            render_kpi_card(
                "Duplicate Rows",
                f"{diff_stats['curr_dup']:,}",
                delta=f"{diff_stats['dup_diff']}" if diff_stats['dup_diff'] != 0 else None,
                delta_text="from original",
                icon="👥"
            )
        with c4:
            render_kpi_card(
                "Data Health Score",
                f"{diff_stats['curr_health']} / 100",
                delta=f"{diff_stats['health_diff']:+d}" if diff_stats['health_diff'] != 0 else None,
                delta_text="health score",
                icon="🛡️"
            )

        st.divider()

        # -------------------------------------------------------------------
        # 2. CLEAR & ACTIONABLE RECOMMENDATIONS (REQUIREMENT 5)
        # -------------------------------------------------------------------
        st.markdown("### 💡 Recommended Cleaning Actions")
        issues = detect_cleaning_issues(curr_df)
        active_issues = [i for i in issues if i["id"] not in st.session_state.dismissed_issues]

        if not active_issues:
            if not session.audit_trail:
                st.success("Your data looks good. No cleaning changes were necessary.")
            else:
                st.success("🎉 All detected quality issues have been addressed! Your dataset is clean and ready.")
        else:
            confirmed_issues = [i for i in active_issues if i["type"] == "Confirmed Issue"]
            review_issues = [i for i in active_issues if i["type"] == "Requires Review"]

            if confirmed_issues:
                st.markdown("#### Confirmed Issues")
                for item in confirmed_issues:
                    render_html(f"""
                    <div class="content-card" style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 16.5px; font-weight: 600; color: #FFFFFF;">{item['name']}</span>
                            <span class="badge-blue">Confirmed Quality Issue</span>
                        </div>
                        <div style="font-size: 14px; color: #94A3B8; margin-bottom: 6px;"><b style="color: #FFFFFF;">Problem:</b> {item['problem']}</div>
                        <div style="font-size: 14px; color: #E2E8F0; margin-bottom: 6px;"><b style="color: #FFFFFF;">Recommended Action:</b> {item['recommended_action']}</div>
                        <div style="font-size: 13.5px; color: #94A3B8; margin-bottom: 8px;"><b style="color: #FFFFFF;">Why:</b> {item['reason']}</div>
                        <div style="font-size: 13px; color: #34D399; font-weight: 600; background: #181B24; border: 1px solid #2E3440; border-radius: 6px; padding: 7px 12px;">Expected outcome: {item['preview']}</div>
                    </div>
                    """)

                    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
                    with btn_col1:
                        if st.button("Apply Fix", key=f"apply_{item['id']}", type="primary"):
                            if item["id"] == "duplicates":
                                apply_remove_duplicates(session)
                            elif item["id"].startswith("missing_"):
                                col = item["column"]
                                strat = "median" if pd.api.types.is_numeric_dtype(curr_df[col]) else "mode"
                                apply_missing_imputation(session, column=col, strategy=strat)
                            elif item["id"].startswith("text_formatting_"):
                                apply_text_standardization(session, column=item["column"], operation="trim_and_title")
                            elif item["id"].startswith("unparsed_date_"):
                                apply_date_parsing(session, column=item["column"])
                            
                            st.session_state.sql_engine.register_table("data", session.current_df)
                            st.rerun()
                    with btn_col2:
                        if st.button("Cancel / Dismiss", key=f"dismiss_{item['id']}"):
                            st.session_state.dismissed_issues.add(item["id"])
                            st.rerun()
                    st.write("")

            if review_issues:
                st.markdown("#### Potential Issues Requiring Human Review")
                st.caption("These items may represent valid business data or unusual extremes. Review before modifying.")
                for item in review_issues:
                    render_html(f"""
                    <div class="content-card" style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 16.5px; font-weight: 600; color: #FFFFFF;">{item['name']}</span>
                            <span class="badge-neutral">Review Recommended</span>
                        </div>
                        <div style="font-size: 14px; color: #94A3B8; margin-bottom: 6px;"><b style="color: #FFFFFF;">Problem:</b> {item['problem']}</div>
                        <div style="font-size: 14px; color: #E2E8F0; margin-bottom: 6px;"><b style="color: #FFFFFF;">Recommended Action:</b> {item['recommended_action']}</div>
                        <div style="font-size: 13.5px; color: #94A3B8; margin-bottom: 8px;"><b style="color: #FFFFFF;">Why:</b> {item['reason']}</div>
                        <div style="font-size: 13px; color: #60A5FA; font-weight: 600; background: #181B24; border: 1px solid #2E3440; border-radius: 6px; padding: 7px 12px;">Expected outcome: {item['preview']}</div>
                    </div>
                    """)

                    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
                    with btn_col1:
                        if st.button("Confirm & Apply", key=f"apply_{item['id']}"):
                            if item["id"].startswith("outliers_"):
                                apply_outlier_handling(session, column=item["column"], method="iqr", action="cap")
                            elif item["id"].startswith("constant_"):
                                apply_drop_column(session, column=item["column"], reason="Zero-variance constant feature")
                            st.session_state.sql_engine.register_table("data", session.current_df)
                            st.rerun()
                    with btn_col2:
                        if st.button("Keep as Is", key=f"dismiss_{item['id']}"):
                            st.session_state.dismissed_issues.add(item["id"])
                            st.rerun()
                    st.write("")

        # -------------------------------------------------------------------
        # 3. MANUAL / CUSTOM CLEANING CONTROLS
        # -------------------------------------------------------------------
        with st.expander("🛠️ Custom / Manual Cleaning Operations", expanded=False):
            st.markdown("Customize specific transformations or fine-tune parameters:")
            op_choice = st.selectbox(
                "Select Cleaning Action:",
                [
                    "Impute Missing Values (Custom Method)",
                    "Remove Duplicate Rows (Custom Keep)",
                    "Standardize Text Formatting",
                    "Handle Numerical Outliers (Custom Threshold)",
                    "Drop Specific Column"
                ]
            )

            if op_choice == "Impute Missing Values (Custom Method)":
                null_cols = [c for c in curr_df.columns if curr_df[c].isnull().sum() > 0]
                if not null_cols:
                    st.info("No columns currently contain missing values.")
                else:
                    sel_col = st.selectbox("Select Column:", null_cols)
                    null_cnt = int(curr_df[sel_col].isnull().sum())
                    is_num = pd.api.types.is_numeric_dtype(curr_df[sel_col])
                    st.write(f"Column **`{sel_col}`** has **{null_cnt}** missing values.")
                    
                    strategies = ["median", "mean", "mode", "custom", "ffill"] if is_num else ["mode", "custom", "ffill"]
                    strat = st.selectbox("Strategy:", strategies)
                    c_val = st.text_input("Custom Value:", value="Unknown") if strat == "custom" else None

                    if st.button("Execute Imputation", type="primary"):
                        success, msg = apply_missing_imputation(session, column=sel_col, strategy=strat, custom_value=c_val)
                        if success:
                            st.session_state.sql_engine.register_table("data", session.current_df)
                            st.success(msg)
                            st.rerun()

            elif op_choice == "Remove Duplicate Rows (Custom Keep)":
                dup_cnt = int(curr_df.duplicated().sum())
                st.write(f"Detected **{dup_cnt}** duplicate rows.")
                k_strat = st.selectbox("Keep Occurrence:", ["first", "last"])
                if st.button("Execute Duplicate Removal", type="primary", disabled=dup_cnt == 0):
                    success, msg = apply_remove_duplicates(session, keep_strategy=k_strat)
                    if success:
                        st.session_state.sql_engine.register_table("data", session.current_df)
                        st.success(msg)
                        st.rerun()

            elif op_choice == "Standardize Text Formatting":
                txt_cols = list(curr_df.select_dtypes(include=["object", "string"]).columns)
                if not txt_cols:
                    st.info("No text columns found.")
                else:
                    s_col = st.selectbox("Text Column:", txt_cols)
                    t_mode = st.selectbox("Operation:", [
                        ("trim_and_title", "Trim Whitespace & Title Case"),
                        ("trim_and_lower", "Trim Whitespace & Lowercase"),
                        ("trim_and_upper", "Trim Whitespace & Uppercase"),
                        ("trim", "Trim Spaces Only")
                    ], format_func=lambda x: x[1])
                    if st.button("Execute Standardization", type="primary"):
                        success, msg = apply_text_standardization(session, column=s_col, operation=t_mode[0])
                        if success:
                            st.session_state.sql_engine.register_table("data", session.current_df)
                            st.success(msg)
                            st.rerun()

            elif op_choice == "Handle Numerical Outliers (Custom Threshold)":
                nums = list(curr_df.select_dtypes(include=["number"]).columns)
                if not nums:
                    st.info("No numerical columns found.")
                else:
                    o_col = st.selectbox("Numerical Column:", nums)
                    o_act = st.radio("Outlier Action:", [
                        ("cap", "Cap / Winsorize (clip values to fences)"),
                        ("flag", "Flag Only (create is_outlier column)"),
                        ("drop", "Drop Rows (remove outlier records)")
                    ], format_func=lambda x: x[1])
                    if st.button("Execute Outlier Handling", type="primary"):
                        success, msg = apply_outlier_handling(session, column=o_col, method="iqr", action=o_act[0])
                        if success:
                            st.session_state.sql_engine.register_table("data", session.current_df)
                            st.success(msg)
                            st.rerun()

            elif op_choice == "Drop Specific Column":
                col_del = st.selectbox("Select Column to Remove:", curr_df.columns)
                del_reason = st.text_input("Reason:", value="Analytical irrelevance or high redundancy")
                if st.button("Drop Column", type="primary"):
                    success, msg = apply_drop_column(session, column=col_del, reason=del_reason)
                    if success:
                        st.session_state.sql_engine.register_table("data", session.current_df)
                        st.success(msg)
                        st.rerun()

        # -------------------------------------------------------------------
        # 4. VIEW CLEANING DETAILS (EXPANDABLE - REQUIREMENTS 2 & 4)
        # -------------------------------------------------------------------
        with st.expander("🔍 View cleaning details", expanded=False):
            st.markdown("#### Three-Part Explanation of Applied Operations")
            if not session.audit_trail:
                st.write("No cleaning operations have been applied yet.")
            else:
                for entry in reversed(session.audit_trail):
                    st.markdown(f"**{entry.issue_name} in {entry.affected_target}** (Step #{entry.step_number})")
                    st.markdown(f"- **What was found:** {entry.what_found}")
                    st.markdown(f"- **What was changed:** {entry.what_changed}")
                    st.markdown(f"- **Why:** {entry.why}")
                    st.caption(f"Method: `{entry.method_details}` | Before: {entry.before_state} ➔ After: {entry.after_state} | Risks: {entry.limitations_risks}")
                    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #D5D7DC;'>", unsafe_allow_html=True)

                st.markdown("#### Full Audit Log Table")
                audit_records = [a.to_dict() for a in session.audit_trail]
                st.dataframe(pd.DataFrame(audit_records), use_container_width=True)

            st.markdown("#### Undo & Reset Options")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                if st.button("↩️ Undo Last Operation", disabled=not session.can_undo(), use_container_width=True):
                    ok, msg = session.undo_last_step()
                    if ok:
                        st.session_state.sql_engine.register_table("data", session.current_df)
                        st.success(msg)
                        st.rerun()
            with r_col2:
                if st.button("🔄 Restore Original Raw Data", use_container_width=True):
                    session.reset_to_original()
                    st.session_state.dismissed_issues = set()
                    st.session_state.sql_engine.register_table("data", session.current_df)
                    st.success("Dataset restored to original raw state.")
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE 5: CHART GUIDE / VISUALIZATION ASSISTANT
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "📊 Chart Guide":
    render_page_header(
        title="Beginner-Friendly Chart Guide & Visual Assistant",
        subtitle="A dataset-aware visualization advisor: pick your analytical goal, receive tailored chart recommendations, and generate verified interactive Plotly graphics with grounded interpretations.",
        badge="Visualization Guide"
    )
    render_workflow_stepper("Chart Guide")

    curr_df = get_current_df()
    if curr_df is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload & Understand** page.")
    else:
        # Step 1: User's Analytical Goal
        st.markdown("### Step 1: What is your analytical goal?")
        st.caption("Select what question or relationship you want to explore:")

        goal_keys = list(ANALYTICAL_GOALS.keys())
        goal_display = [f"{ANALYTICAL_GOALS[k]['icon']} {ANALYTICAL_GOALS[k]['title']} — {ANALYTICAL_GOALS[k]['description']}" for k in goal_keys]
        
        selected_goal_idx = st.selectbox(
            "Choose Analytical Goal:",
            range(len(goal_keys)),
            format_func=lambda i: goal_display[i],
            key="chart_guide_goal_select"
        )
        active_goal_key = goal_keys[selected_goal_idx]
        active_goal = ANALYTICAL_GOALS[active_goal_key]

        # Step 2: Inspect Active Dataset
        schema_info = inspect_dataset_for_visualization(curr_df)
        with st.expander("🔍 Step 2: Dataset Schema Inspection for Visualization", expanded=False):
            sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
            with sc_c1:
                render_kpi_card("Total Rows", f"{schema_info['total_rows']:,}", icon="📏")
            with sc_c2:
                render_kpi_card("Numerical Columns", str(len(schema_info['num_cols'])), icon="📈")
            with sc_c3:
                render_kpi_card("Categorical Columns", str(len(schema_info['cat_cols'])), icon="🏷️")
            with sc_c4:
                render_kpi_card("Date/Time Columns", str(len(schema_info['date_cols'])), icon="📅")
            
            st.caption(f"Low Cardinality (suitable for grouping): `{', '.join(schema_info['low_card_cats']) or 'None'}`")

        # Step 3: Recommend Appropriate Charts
        st.markdown(f"### Step 3: Recommended Charts for: {active_goal['icon']} {active_goal['title']}")
        rec_data = get_chart_recommendations(curr_df, active_goal_key)
        recommendations = rec_data.get("recommendations", [])

        if not recommendations:
            st.info(rec_data.get("message", "No specific recommendations for this goal."))
        else:
            rec_cols = st.columns(len(recommendations))
            for idx, rec in enumerate(recommendations):
                with rec_cols[idx]:
                    is_top = "Highly" in rec.get("badge", "")
                    badge_class = "badge-blue" if is_top else "badge-neutral"
                    render_html(f"""
                    <div class="content-card" style="margin-bottom: 14px; min-height: 230px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-size: 17px; font-weight: 600; color: #FFFFFF;">{rec['chart_type']}</span>
                            <span class="{badge_class}">{rec.get('badge', 'Recommended')}</span>
                        </div>
                        <p style="font-size: 14px; color: #94A3B8; line-height: 1.5; margin-bottom: 12px;">{rec.get('why_useful', '')}</p>
                        <div style="font-size: 13px; color: #E2E8F0; background: #181B24; border: 1px solid #2E3440; border-radius: 8px; padding: 12px 14px; line-height: 1.6;">
                            <div><strong style="color: #FFFFFF;">Axes:</strong> {rec.get('axes_explanation', 'Auto')}</div>
                            <div><strong style="color: #FFFFFF;">Look for:</strong> {rec.get('what_to_look_for', 'Visual patterns')}</div>
                            <div><strong style="color: #FFFFFF;">Avoid when:</strong> {rec.get('when_to_avoid', 'N/A')}</div>
                        </div>
                    </div>
                    """)

        # Step 4: Configure and Generate Real Chart
        st.markdown("### Step 4: Configure & Generate Chart")
        
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        
        # Build available chart types list
        recommended_chart_keys = [r["chart_type"] for r in recommendations]
        all_chart_types = [
            ("Bar Chart", "Bar Chart"),
            ("Grouped Bar Chart", "Grouped Bar Chart"),
            ("Line Chart", "Line Trend Chart"),
            ("Histogram", "Histogram Distribution"),
            ("Box Plot", "Box Plot"),
            ("Scatter Plot", "Scatter Plot"),
            ("Donut Chart", "Donut Composition"),
            ("Correlation Heatmap", "Correlation Heatmap"),
            ("Missingness Bar Chart", "Missing Values Chart")
        ]
        
        # Default chart type to top recommendation if available
        def_chart_idx = 0
        if recommended_chart_keys:
            top_rec = recommended_chart_keys[0]
            for idx_t, (ct_k, _) in enumerate(all_chart_types):
                if ct_k.lower() in top_rec.lower() or top_rec.lower() in ct_k.lower():
                    def_chart_idx = idx_t
                    break

        with cfg_col1:
            sel_chart_type = st.selectbox(
                "Chart Type:",
                all_chart_types,
                index=def_chart_idx,
                format_func=lambda x: x[1],
                key="cg_chart_type"
            )[0]

        # Dynamic axis selections based on chart type
        all_cols = list(curr_df.columns)
        num_cols = schema_info["num_cols"]
        cat_cols = schema_info["cat_cols"]
        dt_cols = schema_info["date_cols"]

        with cfg_col2:
            if sel_chart_type in ["Bar Chart", "Grouped Bar Chart", "Donut Chart"]:
                candidate_x = cat_cols + num_cols if cat_cols else all_cols
                x_col = st.selectbox("Category / X-Axis Column:", candidate_x, key="cg_x_col")
            elif sel_chart_type == "Line Chart":
                candidate_x = dt_cols + num_cols + cat_cols if dt_cols else all_cols
                x_col = st.selectbox("Time / Sequence Column (X):", candidate_x, key="cg_x_col")
            elif sel_chart_type in ["Histogram", "Box Plot"]:
                candidate_x = num_cols if num_cols else all_cols
                x_col = st.selectbox("Numerical Column (X):", candidate_x, key="cg_x_col")
            elif sel_chart_type == "Scatter Plot":
                candidate_x = num_cols if num_cols else all_cols
                x_col = st.selectbox("First Variable (X):", candidate_x, key="cg_x_col")
            else:
                x_col = None
                st.caption("Axis auto-configured for this chart type.")

        with cfg_col3:
            if sel_chart_type in ["Bar Chart", "Grouped Bar Chart"]:
                candidate_y = ["(Record Count)"] + num_cols
                y_sel = st.selectbox("Value / Y-Axis Metric:", candidate_y, key="cg_y_col")
                y_col = None if y_sel == "(Record Count)" else y_sel
            elif sel_chart_type in ["Line Chart", "Scatter Plot"]:
                candidate_y = num_cols if num_cols else all_cols
                y_col = st.selectbox("Second Variable (Y):", candidate_y, key="cg_y_col")
            elif sel_chart_type == "Donut Chart":
                candidate_y = ["(Count of Rows)"] + num_cols
                y_sel = st.selectbox("Metric to Aggregate:", candidate_y, key="cg_y_col")
                y_col = None if y_sel == "(Count of Rows)" else y_sel
            else:
                y_col = None
                st.caption("No secondary Y-axis required.")

        with cfg_col4:
            if sel_chart_type in ["Bar Chart", "Grouped Bar Chart", "Line Chart", "Donut Chart"]:
                agg_options = ["mean", "sum", "median", "count"] if y_col else ["count"]
                agg_func = st.selectbox("Aggregation:", agg_options, key="cg_agg_func")
            else:
                agg_func = "mean"
                st.caption("Default aggregation.")

        # Optional color / grouping
        opt_col1, _ = st.columns([2, 6])
        with opt_col1:
            color_candidates = ["None"] + [c for c in cat_cols if curr_df[c].nunique() <= 10 and c != x_col]
            color_sel = st.selectbox("Optional Grouping / Color:", color_candidates, key="cg_color_col")
            color_col = None if color_sel == "None" else color_sel

        # Generate & Display Chart
        fig, err_msg = generate_guided_chart(
            df=curr_df,
            chart_type=sel_chart_type,
            x_col=x_col,
            y_col=y_col,
            agg_func=agg_func,
            color_col=color_col,
            theme_name=st.session_state.theme
        )

        if err_msg:
            st.error(f"⚠️ {err_msg}")
        elif fig is not None:
            st.plotly_chart(fig, use_container_width=True)

            # Step 5: Interpret Result
            st.markdown("### Step 5: Grounded Analytical Interpretation")
            st.caption("Factual, deterministic observations computed directly from the chart's displayed data:")
            
            insights = derive_chart_insights(
                df=curr_df,
                chart_type=sel_chart_type,
                x_col=x_col,
                y_col=y_col,
                agg_func=agg_func,
                color_col=color_col
            )
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("Chart generated successfully. Adjust axes or groupings above to explore specific feature comparisons.")


# ---------------------------------------------------------------------------
# PAGE 6: SQL ASSISTANT
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "⚡ SQL Assistant":
    render_page_header(
        title="DuckDB Analytical SQL Assistant & Workspace",
        subtitle="High-performance analytical queries executed on in-memory DuckDB with security sandboxing, natural-language translation, and beginner query explanations.",
        badge="SQL Engine"
    )
    render_workflow_stepper("SQL")

    curr_df = get_current_df()
    if curr_df is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        suggested_queries = generate_suggested_queries(curr_df, table_name="data")
        with st.expander("💡 Pre-Built Analytical Query Templates (Adapted to your Schema)", expanded=False):
            s_cols = st.columns(len(suggested_queries[:4]))
            for idx, item in enumerate(suggested_queries[:4]):
                with s_cols[idx]:
                    st.markdown(f"**{item['title']}**")
                    if st.button("Use Template", key=f"tpl_{idx}", use_container_width=True):
                        st.session_state.active_sql_query = item["sql"]
                        st.rerun()

        st.markdown("#### Natural Language to SQL Query:")
        nl_input = st.text_input("Ask a question in plain English (e.g., 'top 5 categories by total revenue' or 'average salary by department'):")
        if st.button("Generate SQL from Question"):
            generated_sql = rule_based_nl_to_sql(nl_input, curr_df, table_name="data")
            st.session_state.active_sql_query = generated_sql

        default_sql = getattr(st.session_state, "active_sql_query", f"SELECT * FROM data LIMIT 10;")
        sql_editor = st.text_area("SQL Query Editor (Table is named 'data'):", value=default_sql, height=140)

        run_col1, run_col2, _ = st.columns([2, 2, 4])
        with run_col1:
            execute_btn = st.button("▶️ Execute Query", type="primary", use_container_width=True)
        with run_col2:
            explain_btn = st.button("📖 Explain Query Clauses", use_container_width=True)

        if explain_btn:
            st.markdown("#### Clause-by-Clause Query Explanation:")
            clauses = explain_sql_query(sql_editor)
            for clause, desc in clauses.items():
                st.markdown(f"- **`{clause}`**: {desc}")

        if execute_btn:
            with st.spinner("Executing DuckDB query in sandbox..."):
                result_df, runtime_ms, err = st.session_state.sql_engine.execute_query(sql_editor)
                if err:
                    st.error(f"❌ Execution Error: {err}")
                elif result_df is not None:
                    st.success(f"✓ Query executed successfully in **{runtime_ms} ms** ({len(result_df):,} rows returned).")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Query Results (CSV)",
                        data=csv_data,
                        file_name="sql_query_results.csv",
                        mime="text/csv"
                    )


# ---------------------------------------------------------------------------
# PAGE 7: AI DATA ANALYST
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "🤖 AI Data Analyst":
    render_page_header(
        title="AI Data Analyst Assistant",
        subtitle="Ask plain-English questions about your dataset. Answers are grounded in exact calculations with zero hallucinated figures.",
        badge="AI Analyst"
    )
    render_workflow_stepper("AI")

    curr_df = get_current_df()
    if curr_df is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        st.markdown("#### Ask a Question About Your Data:")
        st.caption("Quick prompts:")
        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button("Top Categories by Volume", use_container_width=True):
                st.session_state.ai_question = "Which category has the highest values?"
        with q_cols[1]:
            if st.button("Strongest Correlations", use_container_width=True):
                st.session_state.ai_question = "What are the strongest correlations between features?"
        with q_cols[2]:
            if st.button("Outlier & Anomaly Scan", use_container_width=True):
                st.session_state.ai_question = "Are there unusual values or outliers in this dataset?"
        with q_cols[3]:
            if st.button("Numerical Summary", use_container_width=True):
                st.session_state.ai_question = "What is the average summary of numerical columns?"

        default_q = getattr(st.session_state, "ai_question", "Which category has the highest values?")
        user_q = st.text_input("Enter your data question:", value=default_q)

        if st.button("🚀 Analyze Dataset", type="primary"):
            with st.spinner("Analyzing dataset mathematically..."):
                response = query_data_analyst(
                    user_q,
                    curr_df,
                    api_key=st.session_state.api_key,
                    api_provider=st.session_state.api_provider,
                    theme_name=st.session_state.theme
                )

                st.markdown(response["answer"])

                if response.get("chart"):
                    st.plotly_chart(response["chart"], use_container_width=True)

                if response.get("table") is not None:
                    st.dataframe(response["table"], use_container_width=True)

                # Expandable technical calculation details (Requirement 6)
                with st.expander("🔍 How was this calculated?", expanded=False):
                    st.markdown(f"**Methodology:**\n{response.get('calculation_details', 'Calculated directly on dataset.')}")
                    st.markdown(f"**Engine Mode:** `{response.get('mode', 'Deterministic Calculations')}`")
                    if response.get("facts"):
                        st.write("Supporting Metrics:")
                        st.json(response["facts"])


# ---------------------------------------------------------------------------
# PAGE 8: MACHINE LEARNING LAB
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "🧪 Machine Learning Lab":
    render_page_header(
        title="Machine Learning Lab & Predictive Studio",
        subtitle="End-to-end reproducible modeling: task detection, leakage safeguards, preprocessing pipelines, model comparison, and interactive what-if live predictions.",
        badge="ML Modeling"
    )
    render_workflow_stepper("ML")

    curr_df = get_current_df()
    if curr_df is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        # Step 1: Select Target & Auto-Detect Task
        st.markdown("### Step 1: Select Target Feature & Modeling Task")
        target_col = st.selectbox(
            "Choose Target Feature (y):",
            curr_df.columns,
            index=len(curr_df.columns) - 1,
            key="ml_target_select"
        )

        # Clear stale pipeline if target changed
        if st.session_state.ml_pipeline is not None and getattr(st.session_state.ml_pipeline, "target_col", None) != target_col:
            st.session_state.ml_pipeline = None
            if "ml_live_prediction" in st.session_state:
                del st.session_state["ml_live_prediction"]

        inferred_task, task_desc = auto_detect_ml_task(curr_df, target_col)
        
        t_col1, t_col2 = st.columns([3, 2])
        with t_col1:
            st.info(f"**Auto-Detected Task:** `{inferred_task}` — {task_desc}")
            task_choice = st.radio(
                "Confirm or Override Task:",
                ["Classification", "Regression"],
                index=0 if inferred_task == "Classification" else 1,
                horizontal=True,
                key="ml_task_override"
            )
        with t_col2:
            target_series = curr_df[target_col].dropna()
            if task_choice == "Classification":
                u_cnt = target_series.nunique()
                top_classes = target_series.value_counts().head(5).to_dict()
                cls_summary = ", ".join([f"{k}: {v}" for k, v in top_classes.items()])
                st.caption(f"**Target Classes ({u_cnt} distinct):** {cls_summary}")
            else:
                st.caption(f"**Target Range:** Min {target_series.min():.2f} &bull; Median {target_series.median():.2f} &bull; Max {target_series.max():.2f}")

        st.markdown("### Step 2: Feature Selection & Leakage Safeguards")
        candidate_features = [c for c in curr_df.columns if c != target_col]
        
        # Leakage safeguards
        leakage_warnings = check_target_leakage(curr_df, target_col, candidate_features)
        if leakage_warnings:
            for lw in leakage_warnings:
                st.warning(f"⚠️ **Target Leakage Alert:** {lw['warning']}")

        # Smart defaults: exclude IDs, codes, names, and single-value columns
        smart_defaults = [
            c for c in candidate_features
            if "id" not in c.lower() and "code" not in c.lower() and "name" not in c.lower()
            and curr_df[c].nunique() > 1 and curr_df[c].nunique() <= 100
        ]
        if not smart_defaults:
            smart_defaults = candidate_features[:6]

        selected_features = st.multiselect(
            "Select Features (X) to include in model training:",
            candidate_features,
            default=[c for c in smart_defaults if c in candidate_features],
            key=f"ml_features_{target_col}"
        )

        st.markdown("### Step 3: Configure Model Strategy & Test Split")
        m_cfg1, m_cfg2 = st.columns([3, 2])
        
        with m_cfg1:
            avail_models = get_available_models(task_choice)
            train_mode = st.radio(
                "Training Mode:",
                ["Benchmark All Models (Leaderboard Comparison)", "Train Specific Model Only"],
                horizontal=True,
                key="ml_train_mode"
            )
            if train_mode == "Train Specific Model Only":
                chosen_single_model = st.selectbox("Select Model Algorithm:", avail_models, key="ml_single_model_choice")
                models_to_train = [chosen_single_model]
            else:
                models_to_train = None

        with m_cfg2:
            test_pct = st.slider("Train / Test Split (Test %):", min_value=10, max_value=40, value=20, step=5, key="ml_test_split_slider") / 100.0
            st.caption(f"Train on {int((1 - test_pct)*100)}% of valid rows, evaluate on {int(test_pct*100)}% out-of-sample data.")

        # Train button
        if st.button("🚀 Train Model(s) & Evaluate Results", type="primary", disabled=len(selected_features) == 0):
            with st.spinner("Training model pipelines with Scikit-learn..."):
                pipeline = MLTrainingPipeline(
                    df=curr_df,
                    target_col=target_col,
                    feature_cols=selected_features,
                    task_type=task_choice,
                    test_size=test_pct
                )

                prep_ok, prep_err = pipeline.prepare_data()
                if not prep_ok:
                    st.error(f"Data preparation error: {prep_err}")
                else:
                    results = pipeline.train_and_compare(selected_models=models_to_train)
                    if not results:
                        st.error(f"Training failed: {pipeline.model_errors}")
                    else:
                        st.session_state.ml_pipeline = pipeline
                        if "ml_live_prediction" in st.session_state:
                            del st.session_state["ml_live_prediction"]
                        st.success(f"Successfully trained {len(results)} model(s)!")

        # Step 4: Real Model Evaluation & Diagnostics
        pipeline = st.session_state.ml_pipeline
        if pipeline and pipeline.model_results:
            st.divider()
            st.markdown(f"### 🏆 Model Performance Leaderboard ({pipeline.task_type})")
            
            results_df = pd.DataFrame(pipeline.model_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            best_m = pipeline.best_model_name
            if best_m:
                render_html(f"""
                <div style="background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.28); border-radius: 8px; padding: 10px 16px; margin: 10px 0 16px 0; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">🏆</span>
                        <span style="font-size: 13.5px; font-weight: 700; color: #FFFFFF;">Top Performing Model: <b>{best_m}</b></span>
                    </div>
                    <span class="badge-blue">Recommended for Inference</span>
                </div>
                """)

            # Metric Explanation Guide
            with st.expander("📖 Academic Metric Explanation Guide", expanded=False):
                if pipeline.task_type == "Classification":
                    st.markdown("""
                    **Classification Performance Metrics Explained:**
                    - **Accuracy**: Proportion of correct predictions out of all predictions. *Higher is better (1.0 = 100%)*.
                    - **Precision**: Out of all samples predicted positive, how many were truly positive (`TP / (TP + FP)`). Critical when false positives have high costs.
                    - **Recall (Sensitivity)**: Out of all actual positive samples, how many were correctly detected (`TP / (TP + FN)`). Critical when false negatives are dangerous.
                    - **F1-Score**: Harmonic mean of Precision and Recall (`2 * (P * R) / (P + R)`). The most balanced metric for imbalanced classes.
                    """)
                else:
                    st.markdown("""
                    **Regression Performance Metrics Explained:**
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (`|y_true - y_pred|`). Expressed in original target units.
                    - **RMSE (Root Mean Squared Error)**: Penalizes large outlier prediction errors more severely than MAE.
                    - **R² Score (Coefficient of Determination)**: Proportion of variance explained by features. *Higher is better (1.0 = perfect explanation)*.
                    """)

            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                if pipeline.task_type == "Classification":
                    cm_fig = pipeline.get_confusion_matrix_plot(theme_name=st.session_state.theme)
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)
                else:
                    reg_fig = pipeline.get_regression_prediction_plot(theme_name=st.session_state.theme)
                    if reg_fig:
                        st.plotly_chart(reg_fig, use_container_width=True)

            with eval_col2:
                imp_fig = pipeline.get_feature_importance_plot(theme_name=st.session_state.theme)
                if imp_fig:
                    st.plotly_chart(imp_fig, use_container_width=True)

            # Step 5: Interactive Live Prediction Studio (What-If Analysis)
            st.markdown("### 🔮 Interactive What-If Scenario Live Predictor")
            st.caption("Adjust input parameters below to generate live model predictions in real time:")

            trained_options = list(pipeline.trained_models.keys())
            sel_inf_model = st.selectbox(
                "Model to use for inference:",
                trained_options,
                index=trained_options.index(pipeline.best_model_name) if pipeline.best_model_name in trained_options else 0,
                key="ml_inference_model_picker"
            )

            pred_inputs = {}
            p_cols = st.columns(3)
            for idx, f_name in enumerate(pipeline.feature_cols):
                with p_cols[idx % 3]:
                    series = curr_df[f_name].dropna()
                    if pd.api.types.is_numeric_dtype(series):
                        min_v = float(series.min())
                        max_v = float(series.max())
                        med_v = float(series.median())
                        if min_v == max_v:
                            pred_inputs[f_name] = st.number_input(f"{f_name} (constant):", value=min_v, disabled=True, key=f"ml_pred_{f_name}")
                        else:
                            pred_inputs[f_name] = st.slider(f"{f_name}:", min_value=min_v, max_value=max_v, value=med_v, key=f"ml_pred_{f_name}")
                    else:
                        cat_opts = [str(x) for x in series.unique() if pd.notnull(x)][:25]
                        if not cat_opts:
                            cat_opts = ["Unknown"]
                        pred_inputs[f_name] = st.selectbox(f"{f_name}:", cat_opts, key=f"ml_pred_{f_name}")

            if st.button("Generate Live Prediction", type="primary", key="btn_run_ml_prediction"):
                pred_res = pipeline.predict_single_instance(pred_inputs, model_name=sel_inf_model)
                st.session_state.ml_live_prediction = pred_res

            # Display prediction result persistently
            live_pred = st.session_state.get("ml_live_prediction")
            if live_pred:
                if "error" in live_pred:
                    st.error(live_pred["error"])
                else:
                    render_html(f"""
                    <div class="content-card" style="margin-top: 16px; border: 1px solid #2E3440;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 13px; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Live Prediction Result</span>
                            <span class="badge-blue">Model: {live_pred['model_used']}</span>
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin: 4px 0 10px 0;">
                            <span style="font-size: 32px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;">
                                {live_pred['prediction']}
                            </span>
                            <span style="font-size: 14px; color: #94A3B8; font-weight: 500;">(Target: {target_col})</span>
                        </div>
                    </div>
                    """)

                    if "probabilities" in live_pred and live_pred["probabilities"]:
                        st.markdown("##### Class Probabilities Breakdown:")
                        prob_cols = st.columns(len(live_pred["probabilities"]))
                        for p_idx, (c_name, c_prob) in enumerate(live_pred["probabilities"].items()):
                            with prob_cols[p_idx % len(prob_cols)]:
                                render_html(f"""
                                <div style="background: #181B24; border: 1px solid #2E3440; border-radius: 8px; padding: 12px 14px; text-align: center;">
                                    <div style="font-size: 13px; color: #94A3B8; font-weight: 600;">Class: {c_name}</div>
                                    <div style="font-size: 22px; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{c_prob}%</div>
                                    <div style="width: 100%; height: 6px; background: #2E3440; border-radius: 9999px; overflow: hidden; margin-top: 6px;">
                                        <div style="width: {c_prob}%; height: 100%; background: {'#3B82F6' if p_idx == 0 else '#60A5FA'}; border-radius: 9999px;"></div>
                                    </div>
                                </div>
                                """)


# ---------------------------------------------------------------------------
# PAGE 9: REPORTS & EXPORTS
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "📑 Reports & Exports":
    render_page_header(
        title="Automated Reports & Workspace Exports",
        subtitle="Export comprehensive PDF academic dossiers via ReportLab, multi-sheet Excel workbooks, and cleaned CSV datasets.",
        badge="Reporting Stage"
    )
    render_workflow_stepper("Report")

    curr_df = get_current_df()
    session = st.session_state.cleaning_session

    if curr_df is None or session is None:
        st.warning("⚠️ Please upload or select a dataset first in the **Upload Dataset** page.")
    else:
        st.markdown("### Export Cleaned Dataset & Audit Trail:")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            csv_clean = curr_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned CSV",
                data=csv_clean,
                file_name=f"cleaned_{st.session_state.dataset_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_col2:
            if session.audit_trail:
                audit_df = pd.DataFrame([a.to_dict() for a in session.audit_trail])
                audit_csv = audit_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📜 Download Audit Trail (CSV)",
                    data=audit_csv,
                    file_name="cleaning_audit_trail.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("Audit Trail Empty", disabled=True, use_container_width=True)

        with exp_col3:
            ml_res = st.session_state.ml_pipeline.model_results if st.session_state.ml_pipeline else None
            excel_bytes = generate_excel_dossier(
                cleaned_df=curr_df,
                audit_trail=session.audit_trail,
                numerical_summary=get_numerical_summary(curr_df),
                ml_results=ml_res,
                dataset_overview=st.session_state.dataset_overview
            )
            st.download_button(
                "📊 Download Excel Dossier (.xlsx)",
                data=excel_bytes,
                file_name="datapilot_analytical_dossier.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.divider()

        st.markdown("### Academic PDF Analytical Dossier (ReportLab):")
        meta = get_dataset_metadata(curr_df)
        insights = generate_comprehensive_insights(curr_df)
        ml_res = st.session_state.ml_pipeline.model_results if st.session_state.ml_pipeline else None

        pdf_bytes = generate_pdf_report(
            metadata=meta,
            audit_trail=session.audit_trail,
            insights=insights,
            ml_results=ml_res,
            dataset_overview=st.session_state.dataset_overview
        )

        st.download_button(
            "📄 Download Academic PDF Report",
            data=pdf_bytes,
            file_name="DataPilot_AI_Analytical_Report.pdf",
            mime="application/pdf",
            type="primary"
        )

        st.markdown("#### Automated Insights Preview:")
        for ins in insights:
            render_html(f"""
            <div class="content-card" style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 17px; font-weight: 600; color: #FFFFFF;">{ins.get('icon', '💡')} {ins['title']}</span>
                    <span class="badge-blue">{ins['badge']}</span>
                </div>
                <div style="font-size: 14.5px; color: #E2E8F0; margin-bottom: 8px; line-height: 1.55;">{ins['finding']}</div>
                <div style="font-size: 13.5px; color: #94A3B8; margin-bottom: 8px; background: #181B24; border: 1px solid #2E3440; border-radius: 6px; padding: 8px 12px;"><b style="color: #FFFFFF;">Supporting Metric:</b> {ins['supporting_metric']}</div>
                <div style="font-size: 14px; color: #60A5FA; font-weight: 600;"><b style="color: #FFFFFF;">Actionable Takeaway:</b> {ins['takeaway']}</div>
            </div>
            """)


# ---------------------------------------------------------------------------
# PAGE 10: SETTINGS & THEME
# ---------------------------------------------------------------------------

elif st.session_state.current_page == "⚙️ Settings & Theme":
    render_page_header(
        title="Workspace Settings & Preferences",
        subtitle="Configure visual styling themes, API keys, and manage workspace persistence.",
        badge="Configuration"
    )

    st.markdown("### Application Visual Themes")
    st.write("Select a visual styling theme applied across all pages, charts, tables, and KPI cards:")
    
    t_cols = st.columns(len(THEMES))
    for idx, (t_name, t_meta) in enumerate(THEMES.items()):
        with t_cols[idx]:
            is_active = t_name == st.session_state.theme
            render_html(f"""
            <div style="padding: 12px; border-radius: 8px; border: 2px solid {t_meta['accent'] if is_active else t_meta['border_color']}; background: {t_meta['bg_secondary']}; margin-bottom: 8px; text-align: center;">
                <div style="font-weight: 700; font-size: 13px; color: {t_meta['text_primary']};">{t_name}</div>
                <div style="font-size: 10px; color: {t_meta['accent']};">{'● Active' if is_active else ''}</div>
            </div>
            """)
            if st.button("Apply", key=f"theme_{idx}", use_container_width=True):
                st.session_state.theme = t_name
                st.rerun()

    st.divider()

    st.markdown("### Optional LLM API Configuration")
    st.write("DataPilot AI works 100% offline using deterministic computation. You can optionally configure an API key for natural language summary synthesis:")

    provider = st.selectbox("API Provider:", ["Google Gemini", "Groq", "OpenAI Compatible"], index=["Google Gemini", "Groq", "OpenAI Compatible"].index(st.session_state.api_provider))
    api_key_input = st.text_input("API Key:", value=st.session_state.api_key, type="password", help="Never hardcoded; stored strictly in your local session.")

    if st.button("Save API Configuration"):
        st.session_state.api_provider = provider
        st.session_state.api_key = api_key_input
        st.success("API settings saved successfully!")

    st.divider()

    st.markdown("### Workspace Management")
    if st.button("⚠️ Reset Entire Workspace Session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
