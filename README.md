# DataPilot AI — Your All-in-One Data Analyst Workspace

[![Python 3.10 - 3.13](https://img.shields.io/badge/Python-3.10%20--%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20Analytics-FFF000.svg)](https://duckdb.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipelines-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-39%20Passed-brightgreen.svg)]()

> **Academic M.Sc. Data Science Project**  
> An end-to-end analytics workspace that transforms raw datasets into cleaned data, analytical SQL queries, guided visual charts, grounded AI insights, and evaluated machine learning predictions with concise explanations.

---

## 🌟 Executive Summary

In routine data science, practitioners often switch between multiple disjointed tools: spreadsheets for manual inspection, database clients for SQL queries, Jupyter notebooks for modeling, and separate reporting scripts.

**DataPilot AI** unifies this lifecycle into a streamlined, high-performance, beginner-friendly workspace. Operating strictly on actual uploaded data, the application guides users through an 8-stage workflow:

$$\text{Upload \& Understand} \longrightarrow \text{Explore} \longrightarrow \text{Clean (Audited)} \longrightarrow \text{Chart Guide} \longrightarrow \text{SQL} \longrightarrow \text{AI Analyst} \longrightarrow \text{ML Lab} \longrightarrow \text{Export}$$

---

## 🚀 Key Features

### 1. Intelligent Dataset Understanding, Overview & Semantic Column Dictionary
- **Automatic Dataset Understanding**:
  - Dynamically inspects column names, data types, sample values, and statistical distributions upon ingestion.
  - Generates a plain-English explanation of what the dataset appears to represent (e.g., *"This dataset appears to contain employee information, including departments, salaries, joining dates, and work experience. It can be used to explore workforce distribution, salary patterns, and employee experience."*).
  - Never hardcoded — adapts dynamically to any uploaded dataset.
- **Concise Dataset Overview**:
  - Displays high-level information first: dataset size (rows and columns), inferred domain/subject, and counts of numerical, categorical, dates, identifiers, and boolean attributes.
  - Expandable section preserves technical details (memory footprint, deep missingness percentages, paginated data preview).
- **Semantic Column Dictionary**:
  - Explains each column's name, detected programmatic data type, inferred semantic meaning, safe sample value, and missing-value counts.
  - Clearly distinguishes between software-detected programmatic types and AI/heuristic inferred semantic meanings.
  - If a column's meaning is ambiguous, cleanly falls back to: *"Meaning could not be confidently inferred"*.
- **Tailored Dynamic Analysis Suggestions**:
  - Automatically suggests actionable analyses tailored strictly to the uploaded columns: categorical segmentation, chronological time-series trends (only if valid date features exist), numerical correlation, data quality remediation, and predictive machine learning tasks.
- **User-Friendly Customization & Context**:
  - Allows users to correct or edit the dataset description if needed.
  - Enables users to provide optional background context (e.g., *"This dataset contains monthly sales records for my college project"*), tailoring insights without modifying underlying data.
- **Privacy & Offline Safety**:
  - 100% functional offline without an external API.
  - Never transmits raw data rows to external LLMs (sends minimal metadata only when configured).
  - Automatically masks sensitive attributes (tokens, passwords).

### 2. Multi-Format Ingestion & Fast Metadata Profiling
- Ingests **CSV** (auto-delimiter and multi-encoding fallback) and **Multi-sheet Excel (.xlsx, .xls)**.
- Computes structural metrics instantly: memory usage (KB/MB), duplicate rows, missing cell counts, and column type classifications.
- Includes **4 pre-packaged realistic academic datasets**:
  1. *E-Commerce Customer Churn* (1,542 rows — injected with real-world flaws: missing values, mixed date formats, casing discrepancies, extreme outliers).
  2. *Employee HR & Attrition* (Multi-sheet Excel: `Employees` and `Department_Info`).
  3. *Medical Patient Health Risk* (1,200 rows — dual target for Regression and Classification).
  4. *Retail Performance Benchmark* (50,000 rows — verified sub-second DuckDB aggregation).

### 2. Smart Data Cleaning with Simplified Explanations & Audit Trail
- **Compact Summary First**:
  - Displays a clean list of changes right after cleaning:
    - *12 missing values filled in 'Age'*
    - *42 duplicate rows removed*
    - *1 column converted to datetime*
  - If no cleaning was needed, clearly states: *"Your data looks good. No cleaning changes were necessary."*
- **Simple 3-Part Explanations**:
  - **What was found:** Plain-English description of the issue.
  - **What was changed:** Specific action taken and exact number of affected rows/values.
  - **Why:** Short, one-sentence plain-English justification.
- **Actionable Recommendations**:
  - Pre-screen issues with concise problem description, recommended action, short reason, preview of effect, and one-click **Apply Fix** or **Dismiss / Cancel** buttons.
  - Clear separation between **Confirmed Issues** (duplicates, nulls, formatting) and **Items Requiring Review** (outliers, constant columns).
- **Expandable Technical Details ("View cleaning details")**:
  - Keeps full academic audit records (methods, parameters, before/after values, and risk disclosures) accessible in an expandable section to keep the main page clean and easy to scan.
  - Supports single-click **Undo** and **Reset to Original Data**.

### 3. Data Explorer & Profiling
- Interactive exploratory data analysis (EDA):
  - Numerical descriptive statistics (Mean, Median, Std, Quartiles, Skewness, Zeros, IQR Outliers).
  - Categorical cardinality and top-frequency tables.
  - Interactive Plotly correlation heatmaps (Pearson / Spearman).
  - Missingness bar charts.
  - Single-feature distribution deep-dive histograms and boxplots.

### 4. Beginner-Friendly Chart Guide & Visual Assistant
- **5-Step Dataset-Aware Guided Workflow**:
  1. *Understand User Goal*: Select from 6 analytical goals (Compare categories, Changes over time, Distributions, Relationships, Proportions, Data quality).
  2. *Schema Inspection*: Analyzes active columns, numerical vs. categorical counts, dates, and category cardinality.
  3. *Tailored Recommendations*: Generates cards explaining why a chart fits, required columns, axis roles, patterns to look for, and when to avoid.
  4. *Real Plotly Chart Generation*: Interactive Bar, Grouped Bar, Line, Histogram, Box, Scatter, Donut, Heatmap, and Missingness charts.
  5. *100% Grounded Mathematical Interpretation*: Automatically computes factual observations (highest/lowest categories, net change percentages, Pearson correlation coefficients) with zero fabrication.
- **Rule-Based & Offline**: Operates deterministically without external LLM API dependency.

### 5. DuckDB Analytical SQL Assistant & Security Sandbox
- High-performance in-memory analytical SQL powered by **DuckDB**.
- **Security Sandbox**: Strictly enforces read-only queries (`SELECT`, `WITH`). Blocks modifications (`DROP`, `DELETE`, `INSERT`) and unauthorized filesystem access (`read_csv`, `copy`).
- Natural Language to SQL converter.
- Clause-by-clause beginner explanation (explaining `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`).
- Pre-built query templates adapted to your schema.

### 6. Grounded AI Data Analyst with Simplified Output
- Ask analytical questions in plain English (*"Which category contributes most?", "Are there outliers?", "What are the strongest correlations?"*).
- **100% Deterministic Ground Truth**: Computes exact mathematics on Pandas/DuckDB without hallucinations.
- Starts with the single most important finding in bold, followed by concise bullet points with numbers.
- Auto-generates accompanying Plotly charts and data tables alongside text answers.
- Technical methodology and calculation formulas are tucked into an expandable **"How was this calculated?"** section.
- Optional LLM integration (Google Gemini, Groq, OpenAI) for executive summary synthesis.

### 7. Scikit-Learn Machine Learning Lab & Predictive Studio
- **Automatic Task Detection & Strict Validation**:
  - Automatically infers Classification vs. Regression based on target datatype and cardinality.
  - Rejects string targets for Regression with instructional error messages.
  - Rejects continuous float targets with $> 25$ unique values for Classification.
- **Label Encoding**: Encapsulates `LabelEncoder` for string classification targets, preserving original human-readable class labels for evaluation and prediction decoding.
- **Datetime Decomposition**: Automatically decomposes `datetime64` features into `_year`, `_month`, `_day`, and `_dayofweek` numeric columns, preventing pipeline crashes.
- **Data & Target Leakage Prevention**:
  - Flags features with correlation $|r| \ge 0.95$ with the target before training.
  - Strictly fits `ColumnTransformer` (Median Imputation + `StandardScaler`, Mode Imputation + `OneHotEncoder`) *only* on `X_train`.
- **Model Leaderboard**:
  - *Classification*: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (Accuracy, Precision, Recall, F1, Confusion Matrix).
  - *Regression*: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting ($R^2$, MAE, RMSE, Prediction vs. Actual plot).
- **Academic Metric Explanation Guide**: Detailed expander explaining what each metric measures and whether higher or lower values are desirable.
- **Interactive What-If Live Predictor**: Real-time slider and dropdown inputs to test live model predictions and class probabilities, with zero-variance slider crash guards.

### 8. Professional Reports & Exports
- **Academic PDF Dossier**: Formatted PDF generated dynamically via **ReportLab** containing metadata, audit trail, statistical insights, and ML evaluations.
- **Multi-Tab Excel Workbook**: Generated via **OpenPyXL** containing Cleaned Data, Audit Trail, Numerical Profiling, and ML Leaderboard.
- Cleaned CSV export.

### 9. Modern UI Design System with Light-First Theme
- Default **Professional Light** theme with crisp white surfaces, soft neutral background, and restrained blue/indigo accents (`#2563EB`, `#4F46E5`).
- 4 alternative selectable themes (Midnight Dark, Lavender Tech, Ocean Blue, Emerald Growth).
- Minimalist KPI cards, responsive layout, and an 8-stage visual workflow stepper.
- **No Dedicated Dashboard**: All charts remain contextually integrated within their respective analytical sections.

---

## 📂 Project Architecture

```
d:\minor_project\
├── app.py                      # Main Streamlit application & page router (10 pages)
├── requirements.txt            # Compatible Python dependency list
├── run_app.bat                 # One-click Windows launch batch script
├── run_app.ps1                 # Windows PowerShell launch script
├── .env.example                # Placeholder environment configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Complete documentation
├── VIVA_PREPARATION.md         # Academic viva & theory defense guide
├── modules/
│   ├── __init__.py
│   ├── ui_components.py        # Themes, CSS injection, KPI cards, stepper
│   ├── data_loader.py          # CSV & Excel loading, metadata, pagination
│   ├── dataset_explainer.py    # Intelligent understanding, column dictionary & suggestions
│   ├── cleaning_engine.py      # Quality detection, 3-part explanations, audit trail
│   ├── chart_guide.py          # 5-step visual assistant, goal recommendations & insights
│   ├── profiler.py             # Descriptive stats, correlation, distributions
│   ├── sql_engine.py           # DuckDB workspace, security sandbox, explainer
│   ├── ai_analyst.py           # Grounded Q&A, math compute, calculation details
│   ├── insights_engine.py      # Automated data stories, anomaly detection
│   ├── ml_lab.py               # Preprocessing pipelines, models, live predictor
│   └── report_generator.py     # PDF (ReportLab) & Excel (OpenPyXL) dossiers
├── sample_data/
│   ├── generate_samples.py     # Deterministic generator script
│   ├── ecommerce_churn.csv     # Messy customer dataset (1,542 rows)
│   ├── employee_attrition.xlsx # Multi-sheet Excel workbook
│   ├── patient_health_risk.csv # Clinical ML dataset (1,200 rows)
│   └── performance_benchmark_50k.csv # 50,000-row benchmark dataset
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Ingestion & pagination tests
    ├── test_dataset_explainer.py # Intelligent understanding & column dictionary tests
    ├── test_cleaning_engine.py # Transformation, 3-part summaries & audit tests
    ├── test_chart_guide.py     # Goals, recommendations, chart generation & insights
    ├── test_sql_engine.py      # DuckDB execution & sandbox security tests
    ├── test_ml_lab.py          # ML pipelines, target validation & live prediction tests
    └── test_performance_and_reports.py # 50K benchmark & PDF/Excel export tests
```

---

## ⚡ Installation & Run Instructions (Windows)

### Prerequisites
- Python 3.10 to 3.13 installed.
- Pip package manager.

### 1. Clone / Navigate to Project Directory
```powershell
cd d:\minor_project
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Test Suite
Verify that all 27 unit and integration tests pass:
```powershell
python -m pytest -v
```

### 4. Launch Application
You can launch using either method:

**Option A: Using PowerShell / Terminal**
```powershell
python -m streamlit run app.py
```

**Option B: Double-Click Batch File**
- Double-click `run_app.bat` to launch immediately.

The application will start and open in your default browser at:
```
http://localhost:8501
```

---

## ⚙️ Configuration & Optional LLM API Keys

DataPilot AI runs **100% offline** using deterministic local calculations. If you wish to enable optional LLM-enriched natural language executive summaries:

1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Add your API key:
   - `GEMINI_API_KEY=your_key_here` (Recommended)
   - or `GROQ_API_KEY=your_key_here`
   - or configure it dynamically inside the application via **Settings & Theme**.

---

## 📊 Performance Benchmarks

Tested on standard local development hardware (Windows 11, Python 3.13):

| Operation | Dataset Size | Execution Time | Memory Overhead |
|:---|:---:|:---:|:---:|
| CSV Ingestion | 50,000 rows × 9 cols | **0.09s** | ~4.8 MB |
| DuckDB Multi-Group Aggregation | 50,000 rows | **~18 ms** | Sub-millisecond queries |
| Duplicate & Missing Detection | 1,542 rows × 13 cols | **~0.04s** | Negligible |
| Full ML Pipeline (4 Models) | 1,200 rows × 11 cols | **0.82s** | Isolated pipeline |
| Multi-Tab Excel Dossier Export | 1,542 rows + 4 sheets | **0.24s** | 101 KB file |
| Academic PDF Dossier Generation | Full report + tables | **0.31s** | 3.7 KB vector PDF |

---

## 🛡️ Academic Integrity & Limitations

- **No Data Fabrication**: Numerical figures shown in reports, SQL outputs, and AI responses are calculated directly from data using Pandas and DuckDB.
- **Correlation Caveat**: Pairwise correlation coefficients ($r$) explicitly carry disclosures stating that correlation does not imply causation.
- **Outlier Transparency**: Outlier detection relies on reproducible Tukey's IQR or Z-score thresholds with documented upper and lower bounds.
- **Single-User Architecture**: Designed as a standalone academic workstation. Does not include multi-tenant user authentication.

---

## 👥 Author & Academic Context

- **Project**: DataPilot AI — Your All-in-One Data Analyst Workspace
- **Degree**: M.Sc. Data Science
- **Core Competencies Demonstrated**: Data Cleaning & Imputation, Analytical SQL, Exploratory Data Analysis, Interactive Visualizations, Machine Learning Pipelines, and Software Testing.
