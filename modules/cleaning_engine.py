"""
DataPilot AI - Smart Data Cleaning Engine & Audit Trail
Detects data quality flaws, generates transparent recommendations,
executes user-guided transformations, and maintains a strict audit log with undo support.
Includes simplified 3-part explanations (What was found, What was changed, Why).
"""

import copy
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class CleaningAuditEntry:
    """
    Represents a single atomic cleaning operation with both a concise 
    three-part explanation (Found, Action, Why) and full technical details.
    """
    def __init__(
        self,
        step_number: int,
        issue_name: str,
        affected_target: str,
        action_name: str,
        method_details: str,
        rationale: str,
        records_affected: int,
        before_state: str,
        after_state: str,
        limitations_risks: str,
        what_found: str = "",
        what_changed: str = "",
        why: str = ""
    ):
        self.step_number = step_number
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.issue_name = issue_name
        self.affected_target = affected_target
        self.action_name = action_name
        self.method_details = method_details
        self.rationale = rationale
        self.records_affected = records_affected
        self.before_state = before_state
        self.after_state = after_state
        self.limitations_risks = limitations_risks
        
        # Concise three-part explanation fields
        self.what_found = what_found or f"{records_affected} issues detected in {affected_target}."
        self.what_changed = what_changed or f"Applied {action_name} affecting {records_affected} records."
        self.why = why or rationale

    def get_three_part_summary(self) -> Dict[str, str]:
        """Returns the simplified three-part explanation."""
        return {
            "title": f"{self.issue_name} in {self.affected_target}",
            "found": self.what_found,
            "action": self.what_changed,
            "why": self.why
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Step": self.step_number,
            "Timestamp": self.timestamp,
            "Issue": self.issue_name,
            "Target": self.affected_target,
            "What Was Found": self.what_found,
            "What Was Changed": self.what_changed,
            "Why": self.why,
            "Action": self.action_name,
            "Method": self.method_details,
            "Affected Count": self.records_affected,
            "Before": self.before_state,
            "After": self.after_state,
            "Risks & Limitations": self.limitations_risks
        }


class CleaningSession:
    """
    Manages the lifecycle of data cleaning:
    - Preserves immutable raw original DataFrame.
    - Tracks current cleaned DataFrame.
    - Maintains a stack of history snapshots for Undo.
    - Logs comprehensive audit trail entries with concise summaries.
    """
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()
        self.current_df = raw_df.copy()
        self.history_snapshots: List[pd.DataFrame] = [raw_df.copy()]
        self.audit_trail: List[CleaningAuditEntry] = []

    def can_undo(self) -> bool:
        return len(self.history_snapshots) > 1

    def undo_last_step(self) -> Tuple[bool, str]:
        """Reverts the dataset to the snapshot before the most recent transformation."""
        if not self.can_undo():
            return False, "No previous cleaning steps to undo. Already at original dataset."
        
        # Pop current snapshot
        self.history_snapshots.pop()
        self.current_df = self.history_snapshots[-1].copy()
        
        if self.audit_trail:
            last_entry = self.audit_trail.pop()
            msg = f"Successfully reverted Step #{last_entry.step_number} ({last_entry.action_name} on {last_entry.affected_target})."
        else:
            msg = "Reverted to previous state."
            
        return True, msg

    def reset_to_original(self):
        """Restores the initial uploaded dataset and clears history."""
        self.current_df = self.raw_df.copy()
        self.history_snapshots = [self.raw_df.copy()]
        self.audit_trail = []

    def commit_step(self, new_df: pd.DataFrame, audit_entry: CleaningAuditEntry):
        """Saves a new step, snapshots the DataFrame for undo, and records the audit log."""
        self.current_df = new_df.copy()
        self.history_snapshots.append(new_df.copy())
        self.audit_trail.append(audit_entry)

    def get_compact_summary(self) -> List[str]:
        """Returns a concise list of bullet points describing all changes made so far."""
        return [entry.what_changed for entry in self.audit_trail]


# ---------------------------------------------------------------------------
# ISSUE DETECTION ENGINE
# ---------------------------------------------------------------------------

def detect_cleaning_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans dataset and returns structured list of confirmed & potential data quality issues.
    Includes clear recommendation, short reason, preview, and confirmation requirement.
    """
    issues = []
    total_rows = len(df)
    if total_rows == 0:
        return issues

    # 1. Duplicate Rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "id": "duplicates",
            "type": "Confirmed Issue",
            "name": "Duplicate Rows",
            "column": "Entire Dataset",
            "count": dup_count,
            "pct": round((dup_count / total_rows) * 100, 1),
            "problem": f"{dup_count} exact duplicate rows found in dataset.",
            "recommended_action": "Remove duplicate rows, keeping the first occurrence.",
            "reason": "Duplicate records artificially inflate sample sizes and distort averages.",
            "preview": f"Dataset will decrease from {total_rows:,} to {total_rows - dup_count:,} rows.",
            "requires_confirmation": False
        })

    # 2. Missing Values per column
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            null_pct = round((null_count / total_rows) * 100, 1)
            is_num = pd.api.types.is_numeric_dtype(df[col])

            if is_num:
                action = f"Fill all {null_count} missing values using the median value."
                reason = "The median is less affected by extreme values than the mean."
            else:
                action = f"Fill all {null_count} missing values using the most frequent value (mode)."
                reason = "The mode represents the most common observed category."

            issues.append({
                "id": f"missing_{col}",
                "type": "Confirmed Issue",
                "name": f"Missing Values in '{col}'",
                "column": col,
                "count": null_count,
                "pct": null_pct,
                "problem": f"{null_count} values ({null_pct}%) are missing in '{col}'.",
                "recommended_action": action,
                "reason": reason,
                "preview": f"All {null_count} missing cells in '{col}' will be imputed.",
                "requires_confirmation": null_pct > 40.0
            })

    # 3. Inconsistent Text & Leading/Trailing Whitespace
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample_series = df[col].dropna().astype(str)
        if sample_series.empty:
            continue

        has_spaces = sample_series.apply(lambda x: len(x) != len(x.strip())).sum()
        unique_raw = sample_series.unique()
        unique_lowered = set([s.lower().strip() for s in unique_raw])
        has_casing_conflict = len(unique_raw) > len(unique_lowered)

        if has_spaces > 0 or has_casing_conflict:
            affected = max(has_spaces, len(sample_series) - len(unique_lowered))
            issues.append({
                "id": f"text_formatting_{col}",
                "type": "Confirmed Issue",
                "name": f"Inconsistent Text / Spaces in '{col}'",
                "column": col,
                "count": int(affected),
                "pct": round((affected / total_rows) * 100, 1),
                "problem": f"{affected} entries in '{col}' have inconsistent spaces or mixed casing.",
                "recommended_action": "Trim leading/trailing spaces and standardize text to Title Case.",
                "reason": "Ensures identical categories are grouped together properly.",
                "preview": f"Consolidates {len(unique_raw)} raw variations into {len(unique_lowered)} clean categories.",
                "requires_confirmation": False
            })

    # 4. Mixed / Unparsed Date Strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if "date" in col.lower() or "time" in col.lower() or "year" in col.lower() or "created" in col.lower():
            sample = df[col].dropna().astype(str).head(20)
            date_like = sum(1 for val in sample if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", val))
            if date_like >= 5:
                issues.append({
                    "id": f"unparsed_date_{col}",
                    "type": "Confirmed Issue",
                    "name": f"Unparsed Date Strings in '{col}'",
                    "column": col,
                    "count": total_rows,
                    "pct": 100.0,
                    "problem": f"Dates in '{col}' are stored as raw text strings rather than dates.",
                    "recommended_action": "Convert column to standard Datetime format with error validation.",
                    "reason": "Enables chronological sorting, time trends, and date filtering.",
                    "preview": f"Converts '{col}' to datetime64 format.",
                    "requires_confirmation": False
                })

    # 5. Outliers in Numerical Columns (IQR Method)
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = series[(series < lower_fence) | (series > upper_fence)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_pct = round((outlier_count / total_rows) * 100, 1)
                issues.append({
                    "id": f"outliers_{col}",
                    "type": "Requires Review",
                    "name": f"Outliers in '{col}'",
                    "column": col,
                    "count": outlier_count,
                    "pct": outlier_pct,
                    "problem": f"{outlier_count} values ({outlier_pct}%) fall outside normal range [{lower_fence:.1f}, {upper_fence:.1f}].",
                    "recommended_action": "Review values. Choose to cap extreme values, flag them, or keep them.",
                    "reason": "Extreme values may represent real events or data entry errors.",
                    "preview": f"Extreme values range from {series.min():.1f} to {series.max():.1f}.",
                    "requires_confirmation": True
                })

    # 6. Constant / Zero-Variance Columns
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=False)
        if unique_vals <= 1:
            issues.append({
                "id": f"constant_{col}",
                "type": "Requires Review",
                "name": f"Constant Column '{col}'",
                "column": col,
                "count": total_rows,
                "pct": 100.0,
                "problem": f"Contains only 1 unique value ('{df[col].iloc[0]}') across all rows.",
                "recommended_action": "Remove this column from the dataset.",
                "reason": "Columns with no variation carry no useful analytical or predictive information.",
                "preview": f"Will remove column '{col}', reducing column count by 1.",
                "requires_confirmation": True
            })

    return issues


# ---------------------------------------------------------------------------
# ATOMIC DATA TRANSFORMATIONS WITH THREE-PART AUDIT TRAIL
# ---------------------------------------------------------------------------

def apply_remove_duplicates(
    session: CleaningSession,
    subset_columns: Optional[List[str]] = None,
    keep_strategy: str = "first"
) -> Tuple[bool, str]:
    """Removes duplicate rows with concise 3-part logging."""
    df = session.current_df
    initial_rows = len(df)
    
    dup_mask = df.duplicated(subset=subset_columns, keep=keep_strategy)
    removed_count = int(dup_mask.sum())
    
    if removed_count == 0:
        return False, "No duplicate rows detected matching criteria."
        
    cleaned_df = df[~dup_mask].copy().reset_index(drop=True)
    step_num = len(session.audit_trail) + 1
    target_desc = f"Subset ({', '.join(subset_columns)})" if subset_columns else "All Columns"
    
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name="Duplicate Rows",
        affected_target=target_desc,
        action_name="Remove Duplicate Records",
        method_details=f"drop_duplicates(keep='{keep_strategy}', subset={subset_columns})",
        rationale="Exact duplicate rows represent redundant records that bias calculations.",
        records_affected=removed_count,
        before_state=f"{initial_rows} total rows (including {removed_count} duplicates)",
        after_state=f"{len(cleaned_df)} unique rows remaining",
        limitations_risks="Assumes duplicates are uninformative repeats.",
        what_found=f"{removed_count} duplicate rows were found in the dataset.",
        what_changed=f"{removed_count} duplicate rows removed (kept {keep_strategy} occurrence).",
        why="Duplicate rows artificially inflate sample sizes and distort averages."
    )
    
    session.commit_step(cleaned_df, audit_entry)
    return True, f"Successfully removed {removed_count} duplicate rows."


def apply_missing_imputation(
    session: CleaningSession,
    column: str,
    strategy: str = "median",
    custom_value: Any = None
) -> Tuple[bool, str]:
    """Imputes missing values using Mean, Median, Mode, or Custom Value with 3-part logging."""
    df = session.current_df
    if column not in df.columns:
        return False, f"Column '{column}' does not exist in dataset."
        
    missing_count = int(df[column].isnull().sum())
    if missing_count == 0:
        return False, f"Column '{column}' has zero missing values."

    cleaned_df = df.copy()
    col_series = cleaned_df[column]
    
    if strategy == "median":
        if not pd.api.types.is_numeric_dtype(col_series):
            return False, f"Median imputation cannot be applied to non-numeric column '{column}'."
        fill_val = col_series.median()
        method_desc = f"Median imputation (fill value = {fill_val:.2f})"
        why_sentence = "The median is less affected by extreme values than the mean."
        what_changed_sentence = f"{missing_count} missing values in '{column}' filled using median ({fill_val:.2f})."
        rationale = "The median is robust against extreme values, preserving central tendency."
        risks = "Reduces variance of the feature."

    elif strategy == "mean":
        if not pd.api.types.is_numeric_dtype(col_series):
            return False, f"Mean imputation cannot be applied to non-numeric column '{column}'."
        fill_val = col_series.mean()
        method_desc = f"Mean imputation (fill value = {fill_val:.2f})"
        why_sentence = "Mean imputation preserves the arithmetic average of the column."
        what_changed_sentence = f"{missing_count} missing values in '{column}' filled using mean ({fill_val:.2f})."
        rationale = "Mean imputation preserves the sum and arithmetic average of the column."
        risks = "Distorted if extreme outliers exist."

    elif strategy == "mode":
        modes = col_series.mode()
        fill_val = modes.iloc[0] if not modes.empty else "Unknown"
        method_desc = f"Mode imputation (fill value = '{fill_val}')"
        why_sentence = "The mode represents the most common observed category."
        what_changed_sentence = f"{missing_count} missing values in '{column}' filled using mode ('{fill_val}')."
        rationale = "Mode represents the most frequent observed category."
        risks = "Can artificially inflate the dominant category."

    elif strategy == "custom":
        fill_val = custom_value
        method_desc = f"Custom value imputation (fill value = '{fill_val}')"
        why_sentence = "Replaces missing values with a designated default label."
        what_changed_sentence = f"{missing_count} missing values in '{column}' filled with '{fill_val}'."
        rationale = "Replaces unknown values with a domain-specific default."
        risks = "Creates a distinct synthetic value."

    elif strategy == "ffill":
        cleaned_df[column] = cleaned_df[column].ffill().bfill()
        fill_val = "Forward/Backward Fill"
        method_desc = "Time-series Forward/Backward Fill"
        why_sentence = "Propagates neighboring values in time-ordered data."
        what_changed_sentence = f"{missing_count} missing values in '{column}' filled using neighboring values."
        rationale = "Propagates the last valid observation forward."
        risks = "Only valid if records are strictly ordered in time."

    else:
        return False, f"Unknown imputation strategy: {strategy}"

    if strategy != "ffill":
        cleaned_df[column] = cleaned_df[column].fillna(fill_val)

    step_num = len(session.audit_trail) + 1
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name="Missing Values",
        affected_target=f"Column '{column}'",
        action_name=f"Impute Missing with {strategy.capitalize()}",
        method_details=method_desc,
        rationale=rationale,
        records_affected=missing_count,
        before_state=f"{missing_count} missing values in '{column}'",
        after_state=f"0 missing values in '{column}'",
        limitations_risks=risks,
        what_found=f"{missing_count} values were missing in '{column}'.",
        what_changed=what_changed_sentence,
        why=why_sentence
    )
    
    session.commit_step(cleaned_df, audit_entry)
    return True, f"Successfully filled {missing_count} missing values in '{column}'."


def apply_drop_missing(
    session: CleaningSession,
    axis: str = "rows",
    column: Optional[str] = None
) -> Tuple[bool, str]:
    """Drops rows or columns containing missing values with 3-part logging."""
    df = session.current_df
    initial_rows, initial_cols = len(df), len(df.columns)
    cleaned_df = df.copy()

    if axis == "rows":
        if column:
            missing_count = int(cleaned_df[column].isnull().sum())
            if missing_count == 0:
                return False, f"Column '{column}' has no missing values."
            cleaned_df = cleaned_df.dropna(subset=[column]).reset_index(drop=True)
            records_affected = missing_count
            what_found = f"{missing_count} rows were missing values in '{column}'."
            what_changed = f"{missing_count} incomplete rows removed."
            why = f"Removes records that cannot be analyzed without '{column}'."
        else:
            cleaned_df = cleaned_df.dropna().reset_index(drop=True)
            records_affected = initial_rows - len(cleaned_df)
            what_found = f"{records_affected} rows contained at least one missing cell."
            what_changed = f"{records_affected} incomplete rows removed."
            why = "Ensures only completely observed records are kept."

        rationale = "Removed incomplete rows."
        risks = "Reduces sample size and may introduce selection bias."
        action_name = "Drop Incomplete Rows"
        target_name = f"Rows (subset='{column}')" if column else "All Incomplete Rows"

    elif axis == "columns":
        if not column or column not in cleaned_df.columns:
            return False, f"Column '{column}' does not exist."
        missing_count = int(cleaned_df[column].isnull().sum())
        cleaned_df = cleaned_df.drop(columns=[column])
        records_affected = missing_count
        action_name = f"Drop Column '{column}'"
        target_name = f"Column '{column}'"
        what_found = f"Column '{column}' had {missing_count} missing values."
        what_changed = f"Column '{column}' removed from the dataset."
        why = "Removes a feature with excessive missing values or little analytical value."
        rationale = f"Dropped column '{column}'."
        risks = "Permanently removes all observed data in this column."

    else:
        return False, f"Invalid axis: {axis}"

    step_num = len(session.audit_trail) + 1
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name="Missing Values Removal",
        affected_target=target_name,
        action_name=action_name,
        method_details=f"dropna(axis='{axis}', subset={column})",
        rationale=rationale,
        records_affected=records_affected,
        before_state=f"{initial_rows} rows, {initial_cols} columns",
        after_state=f"{len(cleaned_df)} rows, {len(cleaned_df.columns)} columns",
        limitations_risks=risks,
        what_found=what_found,
        what_changed=what_changed,
        why=why
    )
    
    session.commit_step(cleaned_df, audit_entry)
    return True, f"Successfully executed: {what_changed}."


def apply_text_standardization(
    session: CleaningSession,
    column: str,
    operation: str = "trim_and_title"
) -> Tuple[bool, str]:
    """Standardizes text formatting and casing with 3-part logging."""
    df = session.current_df
    if column not in df.columns:
        return False, f"Column '{column}' does not exist."

    cleaned_df = df.copy()
    series = cleaned_df[column].astype(str)
    
    if operation == "trim":
        new_series = series.str.strip()
    elif operation == "trim_and_title":
        new_series = series.str.strip().str.title()
    elif operation == "trim_and_lower":
        new_series = series.str.strip().str.lower()
    elif operation == "trim_and_upper":
        new_series = series.str.strip().str.upper()
    elif operation == "strip_special":
        new_series = series.apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).strip())
    else:
        return False, f"Unknown text operation: {operation}"

    new_series = new_series.where(df[column].notnull(), np.nan)
    changed_mask = (df[column] != new_series) & (df[column].notnull())
    changed_count = int(changed_mask.sum())
    cleaned_df[column] = new_series
    
    step_num = len(session.audit_trail) + 1
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name="Text Formatting & Whitespace",
        affected_target=f"Column '{column}'",
        action_name=f"Standardize Text ({operation})",
        method_details=f"String standardization: {operation}",
        rationale="Eliminates spacing and casing variations.",
        records_affected=changed_count,
        before_state=f"{df[column].nunique()} unique categories",
        after_state=f"{cleaned_df[column].nunique()} standardized categories",
        limitations_risks="May reformat specialized acronyms.",
        what_found=f"{changed_count} values in '{column}' had inconsistent spacing or casing.",
        what_changed=f"{changed_count} values in '{column}' standardized ({operation.replace('_', ' ')}).",
        why="Ensures identical categories are grouped together properly."
    )
    
    session.commit_step(cleaned_df, audit_entry)
    return True, f"Standardized {changed_count} text records in '{column}'."


def apply_date_parsing(
    session: CleaningSession,
    column: str,
    date_format: Optional[str] = None
) -> Tuple[bool, str]:
    """Parses text to datetime with 3-part logging."""
    df = session.current_df
    if column not in df.columns:
        return False, f"Column '{column}' not found."

    cleaned_df = df.copy()
    try:
        converted = pd.to_datetime(cleaned_df[column], format=date_format, errors="coerce")
        invalid_dates_count = int((cleaned_df[column].notnull()) & (converted.isnull()).sum())
        cleaned_df[column] = converted
        
        step_num = len(session.audit_trail) + 1
        audit_entry = CleaningAuditEntry(
            step_number=step_num,
            issue_name="Date Formatting",
            affected_target=f"Column '{column}'",
            action_name="Parse Column to Datetime",
            method_details=f"to_datetime(errors='coerce')",
            rationale="Allows time-based trends and date filtering.",
            records_affected=len(cleaned_df),
            before_state="Data type: object/string",
            after_state=f"Data type: datetime ({invalid_dates_count} invalid values set to null)",
            limitations_risks="Ambiguous date formats may rely on default parsing heuristics.",
            what_found=f"Column '{column}' was stored as text strings rather than dates.",
            what_changed=f"Column '{column}' converted to datetime format ({invalid_dates_count} invalid dates set to null).",
            why="Enables chronological sorting, time-series analysis, and date filtering."
        )
        
        session.commit_step(cleaned_df, audit_entry)
        return True, f"Converted '{column}' to datetime ({invalid_dates_count} invalid values set to null)."
    except Exception as e:
        return False, f"Failed to parse dates: {str(e)}"


def apply_outlier_handling(
    session: CleaningSession,
    column: str,
    method: str = "iqr",
    action: str = "cap",
    threshold: float = 1.5
) -> Tuple[bool, str]:
    """Handles numerical outliers with 3-part logging."""
    df = session.current_df
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return False, f"Column '{column}' is not a valid numeric column."

    cleaned_df = df.copy()
    series = cleaned_df[column].dropna()
    
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)
    else:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return False, "Column has zero standard deviation."
        z_scores = (cleaned_df[column] - mean) / std
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outlier_mask = z_scores.abs() > threshold

    outlier_count = int(outlier_mask.sum())
    if outlier_count == 0:
        return False, f"No outliers detected in '{column}' using {method.upper()}."

    if action == "flag":
        flag_col = f"is_outlier_{column}"
        cleaned_df[flag_col] = outlier_mask.astype(int)
        method_desc = f"Flagged outliers into boolean column '{flag_col}'"
        what_found = f"{outlier_count} values in '{column}' were outside normal range [{lower_bound:.1f}, {upper_bound:.1f}]."
        what_changed = f"{outlier_count} unusual values flagged in new column '{flag_col}'."
        why = "Flags unusual values for inspection without modifying raw data."

    elif action == "cap":
        cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
        method_desc = f"Capped values to bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
        what_found = f"{outlier_count} values in '{column}' were outside normal range [{lower_bound:.1f}, {upper_bound:.1f}]."
        what_changed = f"{outlier_count} extreme values capped to boundaries [{lower_bound:.1f}, {upper_bound:.1f}]."
        why = "Prevents extreme values from distorting models while keeping all rows."

    elif action == "drop":
        cleaned_df = cleaned_df[~outlier_mask].reset_index(drop=True)
        method_desc = f"Dropped rows where '{column}' is outside bounds"
        what_found = f"{outlier_count} values in '{column}' were outside normal range [{lower_bound:.1f}, {upper_bound:.1f}]."
        what_changed = f"{outlier_count} rows with extreme values removed."
        why = "Excludes extreme anomalies from downstream statistical models."

    else:
        return False, f"Unknown action: {action}"

    step_num = len(session.audit_trail) + 1
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name=f"Outliers in '{column}'",
        affected_target=f"Column '{column}'",
        action_name=f"Handle Outliers ({action.capitalize()})",
        method_details=method_desc,
        rationale=f"Handled outliers using {action}.",
        records_affected=outlier_count,
        before_state=f"{len(df)} rows",
        after_state=f"{len(cleaned_df)} rows",
        limitations_risks="May modify valid extreme values.",
        what_found=what_found,
        what_changed=what_changed,
        why=why
    )

    session.commit_step(cleaned_df, audit_entry)
    return True, f"Successfully executed: {what_changed}."


def apply_drop_column(session: CleaningSession, column: str, reason: str = "Unneeded or constant column") -> Tuple[bool, str]:
    """Drops a column completely with 3-part logging."""
    df = session.current_df
    if column not in df.columns:
        return False, f"Column '{column}' does not exist."

    cleaned_df = df.drop(columns=[column])
    step_num = len(session.audit_trail) + 1
    
    audit_entry = CleaningAuditEntry(
        step_number=step_num,
        issue_name="Column Removal",
        affected_target=f"Column '{column}'",
        action_name="Drop Column",
        method_details=f"drop(columns=['{column}'])",
        rationale=reason,
        records_affected=len(df),
        before_state=f"{len(df.columns)} columns",
        after_state=f"{len(cleaned_df.columns)} columns",
        limitations_risks="Discards any potential signal in this column.",
        what_found=f"Column '{column}' was identified as unneeded or constant.",
        what_changed=f"Column '{column}' removed from the dataset.",
        why="Removes unneeded features to simplify analysis and reduce clutter."
    )
    
    session.commit_step(cleaned_df, audit_entry)
    return True, f"Successfully removed column '{column}'."


def get_cleaning_comparison_stats(raw_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes side-by-side comparative health and structural metrics."""
    raw_rows, raw_cols = len(raw_df), len(raw_df.columns)
    curr_rows, curr_cols = len(current_df), len(current_df.columns)

    raw_missing = int(raw_df.isnull().sum().sum())
    curr_missing = int(current_df.isnull().sum().sum())

    raw_dup = int(raw_df.duplicated().sum())
    curr_dup = int(current_df.duplicated().sum())

    raw_cells = max(1, raw_rows * raw_cols)
    curr_cells = max(1, curr_rows * curr_cols)

    raw_penalty = (raw_missing / raw_cells * 100 * 1.5) + (raw_dup / max(1, raw_rows) * 100 * 2.0)
    curr_penalty = (curr_missing / curr_cells * 100 * 1.5) + (curr_dup / max(1, curr_rows) * 100 * 2.0)

    raw_health = max(10, min(100, int(100 - raw_penalty)))
    curr_health = max(10, min(100, int(100 - curr_penalty)))

    return {
        "raw_rows": raw_rows,
        "curr_rows": curr_rows,
        "rows_diff": curr_rows - raw_rows,
        "raw_cols": raw_cols,
        "curr_cols": curr_cols,
        "cols_diff": curr_cols - raw_cols,
        "raw_missing": raw_missing,
        "curr_missing": curr_missing,
        "missing_diff": curr_missing - raw_missing,
        "raw_dup": raw_dup,
        "curr_dup": curr_dup,
        "dup_diff": curr_dup - raw_dup,
        "raw_health": raw_health,
        "curr_health": curr_health,
        "health_diff": curr_health - raw_health
    }
