"""
DataPilot AI - Report & Export Generator
Generates academic PDF analytical dossiers via ReportLab,
multi-tab Excel workbooks via OpenPyXL, and CSV exports.
Includes dataset overview and column dictionary summaries.
"""

import io
from typing import Dict, List, Any, Optional
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(
    metadata: Dict[str, Any],
    audit_trail: List[Any],
    insights: List[Dict[str, Any]],
    ml_results: Optional[List[Dict[str, Any]]] = None,
    dataset_overview: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Compiles an academic PDF dossier containing dataset metadata, 
    dataset overview explanation, cleaning audit log, statistical insights, and machine learning evaluations.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1E293B")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#64748B")
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # 1. Header & Title
    story.append(Paragraph("DataPilot AI — Analytical Dossier", title_style))
    story.append(Paragraph("Automated Data Science & Audit Trail Report | Academic M.Sc. Project", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#38BDF8"), spaceBefore=10, spaceAfter=12))

    # 2. Dataset Overview
    story.append(Paragraph("1. Dataset Overview & Structural Summary", h2_style))
    if dataset_overview and dataset_overview.get("description"):
        story.append(Paragraph(f"<b>About this dataset:</b> {dataset_overview['description']}", body_style))
        story.append(Spacer(1, 8))

    meta_data = [
        ["Total Rows", str(metadata.get("total_rows", "N/A")), "Total Columns", str(metadata.get("total_cols", "N/A"))],
        ["Memory Footprint", str(metadata.get("memory_str", "N/A")), "Data Health Score", f"{metadata.get('health_score', 'N/A')} / 100"],
        ["Duplicate Rows", str(metadata.get("duplicate_rows", "0")), "Missing Cells", f"{metadata.get('total_missing_cells', '0')} ({metadata.get('missing_pct', '0')}%)"]
    ]
    meta_table = Table(meta_data, colWidths=[120, 140, 120, 140])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Smart Data Cleaning Audit Trail
    story.append(Paragraph("2. Smart Data Cleaning Audit Trail", h2_style))
    if audit_trail:
        audit_rows = [["Step", "Target", "Action", "Records", "Rationale"]]
        for entry in audit_trail[:10]:
            audit_rows.append([
                str(entry.step_number),
                entry.affected_target[:18],
                entry.action_name[:20],
                str(entry.records_affected),
                entry.rationale[:45] + "..." if len(entry.rationale) > 45 else entry.rationale
            ])
        
        audit_table = Table(audit_rows, colWidths=[35, 95, 110, 50, 230])
        audit_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(audit_table)
    else:
        story.append(Paragraph("No cleaning transformations applied yet. Dataset is in raw uploaded state.", body_style))
    story.append(Spacer(1, 10))

    # 4. Statistical Insights & Anomalies
    story.append(Paragraph("3. Executive Data Insights & Methodologies", h2_style))
    for ins in insights[:4]:
        ins_p = f"<b>[{ins.get('badge')}] {ins.get('title')}</b><br/>{ins.get('finding')}<br/><i>Method: {ins.get('methodology')}</i>"
        story.append(Paragraph(ins_p, body_style))
        story.append(Spacer(1, 5))

    # 5. Machine Learning Evaluation (if available)
    if ml_results:
        story.append(Paragraph("4. Machine Learning Model Leaderboard", h2_style))
        headers = list(ml_results[0].keys())
        ml_rows = [headers]
        for m in ml_results:
            ml_rows.append([str(m.get(h, "")) for h in headers])

        col_w = 520 / len(headers)
        ml_table = Table(ml_rows, colWidths=[col_w] * len(headers))
        ml_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ml_table)

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=8, spaceAfter=6))
    story.append(Paragraph("Generated dynamically by DataPilot AI. Operations adhere to academic reproducibility standards.", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_excel_dossier(
    cleaned_df: pd.DataFrame,
    audit_trail: List[Any],
    numerical_summary: Optional[pd.DataFrame] = None,
    ml_results: Optional[List[Dict[str, Any]]] = None,
    dataset_overview: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Exports a multi-tab Excel workbook containing:
    1. Cleaned Data
    2. Column Dictionary (if available)
    3. Audit Trail
    4. Numerical Profiles
    5. ML Leaderboard (if present)
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        cleaned_df.head(50000).to_excel(writer, sheet_name="Cleaned_Data", index=False)

        if dataset_overview and dataset_overview.get("column_dictionary"):
            col_dict_df = pd.DataFrame(dataset_overview["column_dictionary"])
            col_dict_df.to_excel(writer, sheet_name="Column_Dictionary", index=False)

        if audit_trail:
            audit_dicts = [a.to_dict() for a in audit_trail]
            pd.DataFrame(audit_dicts).to_excel(writer, sheet_name="Cleaning_Audit_Trail", index=False)

        if numerical_summary is not None and not numerical_summary.empty:
            numerical_summary.to_excel(writer, sheet_name="Data_Profiling", index=False)

        if ml_results:
            pd.DataFrame(ml_results).to_excel(writer, sheet_name="ML_Leaderboard", index=False)

    buffer.seek(0)
    return buffer.getvalue()
