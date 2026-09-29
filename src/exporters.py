"""Export helpers for the traceability matrix, coverage gaps and match
details tables (CSV and Excel)."""
from __future__ import annotations

import io
from typing import Dict

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframes_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple dataframes to a single .xlsx workbook, one sheet each."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]  # Excel sheet name length limit
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()
