import streamlit as st

from src.exporters import dataframe_to_csv_bytes, dataframes_to_excel_bytes
from src.state import init_session_state
from src.ui_components import guard_no_analysis, style_status_column
from src.utils import STATUS_ORDER

init_session_state()

st.title("🧷 Traceability Matrix")
st.caption(
    "Many-to-many mapping: each requirement can show several candidate test "
    "cases, and a test case can appear against several requirements."
)

if guard_no_analysis():
    st.stop()

matrix = st.session_state["traceability_matrix"]
requirements = st.session_state["requirements_clean"]

col1, col2, col3 = st.columns(3)
with col1:
    req_options = ["All"] + sorted(matrix["requirement_id"].unique().tolist())
    req_filter = st.selectbox("Requirement ID", req_options)
with col2:
    status_options = ["All"] + STATUS_ORDER
    status_filter = st.selectbox("Status", status_options)
with col3:
    priority_options = ["All"] + sorted(matrix["priority"].dropna().unique().tolist())
    priority_filter = st.selectbox("Priority", priority_options)

min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)

filtered = matrix.copy()
if req_filter != "All":
    filtered = filtered[filtered["requirement_id"] == req_filter]
if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]
if priority_filter != "All":
    filtered = filtered[filtered["priority"] == priority_filter]
filtered = filtered[filtered["final_score"] >= min_score]

st.caption(f"Showing {len(filtered)} of {len(matrix)} mapping rows.")

display_cols = [
    "requirement_id",
    "requirement_title",
    "test_case_id",
    "test_case_title",
    "final_score",
    "status",
    "priority",
]
st.dataframe(
    style_status_column(filtered[display_cols].rename(
        columns={
            "requirement_id": "Requirement ID",
            "requirement_title": "Requirement",
            "test_case_id": "Test Case ID",
            "test_case_title": "Test Case",
            "final_score": "Score",
            "status": "status",
            "priority": "Priority",
        }
    ), status_col="status").format({"Score": "{:.2f}"}),
    width='stretch',
    hide_index=True,
)

st.divider()
st.subheader("Export")

exp1, exp2 = st.columns(2)
with exp1:
    st.download_button(
        "⬇️ Download traceability matrix (CSV)",
        data=dataframe_to_csv_bytes(matrix),
        file_name="traceability_matrix.csv",
        mime="text/csv",
    )
with exp2:
    excel_bytes = dataframes_to_excel_bytes(
        {
            "Traceability Matrix": matrix,
            "Requirements": requirements,
            "Test Cases": st.session_state["test_cases_clean"],
        }
    )
    st.download_button(
        "⬇️ Download full workbook (XLSX)",
        data=excel_bytes,
        file_name="traceability_workbook.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
