import streamlit as st

from src.state import init_session_state
from src.ui_components import guard_no_data

init_session_state()

st.title("🧪 Test Cases")

if guard_no_data():
    st.stop()

test_cases = st.session_state["test_cases_clean"]
warnings = st.session_state.get("test_cases_warnings", [])

if warnings:
    with st.expander(f"⚠️ {len(warnings)} data-quality note(s) from validation", expanded=False):
        for w in warnings:
            st.warning(w)

st.caption(f"{len(test_cases)} test case(s) after cleaning and de-duplication.")

priorities = ["All"] + sorted(test_cases["priority"].unique().tolist())
selected_priority = st.selectbox("Filter by priority", priorities)

display_df = test_cases
if selected_priority != "All":
    display_df = display_df[display_df["priority"] == selected_priority]

if st.session_state.get("analysis_has_run"):
    match_table = st.session_state["match_table"]
    strong_threshold = st.session_state["strong_threshold"]
    partial_threshold = st.session_state["partial_threshold"]

    # How many requirements does each test case meaningfully cover?
    relevant = match_table[match_table["final_score"] >= partial_threshold]
    usage_counts = (
        relevant.groupby("test_case_id")["requirement_id"].nunique().rename("requirements_covered")
    )
    display_df = display_df.merge(usage_counts, on="test_case_id", how="left")
    display_df["requirements_covered"] = display_df["requirements_covered"].fillna(0).astype(int)
    cols = ["test_case_id", "title", "description", "priority", "requirements_covered"]
    st.caption(
        "`requirements_covered`: number of requirements for which this test case "
        "scores at or above the Partial threshold."
    )
else:
    st.info("Run the analysis on the Data Upload page to see requirement coverage per test case.")
    cols = ["test_case_id", "title", "description", "priority"]

st.dataframe(display_df[cols], width='stretch', hide_index=True)
