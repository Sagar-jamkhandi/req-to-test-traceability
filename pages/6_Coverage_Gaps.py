import streamlit as st

from src.exporters import dataframe_to_csv_bytes
from src.state import init_session_state
from src.suggestions import generate_suggestions
from src.ui_components import guard_no_analysis, status_badge, style_status_column
from src.utils import STATUS_MISSING, STATUS_PARTIAL

init_session_state()

st.title("🕳️ Coverage Gaps")
st.caption(
    "Requirements with partial or missing coverage, sorted by priority then "
    "by lowest score — the requirements to look at first."
)

if guard_no_analysis():
    st.stop()

gaps = st.session_state["coverage_gaps"]
requirements = st.session_state["requirements_clean"]

if gaps.empty:
    st.success("No coverage gaps found at the current thresholds — every requirement has at least a Strong match.")
    st.stop()

status_filter = st.multiselect(
    "Show statuses", options=[STATUS_MISSING, STATUS_PARTIAL], default=[STATUS_MISSING, STATUS_PARTIAL]
)
filtered = gaps[gaps["status"].isin(status_filter)] if status_filter else gaps

st.caption(f"{len(filtered)} requirement(s) with a coverage gap.")

display = filtered.rename(
    columns={
        "requirement_id": "Requirement ID",
        "requirement": "Requirement",
        "priority": "Priority",
        "best_match_id": "Best Match",
        "best_match_title": "Best Match Title",
        "similarity": "Similarity",
        "status": "status",
        "recommended_action": "Recommended Action",
    }
)
st.dataframe(
    style_status_column(display, status_col="status").format({"Similarity": "{:.2f}"}),
    width='stretch',
    hide_index=True,
)

st.divider()
st.subheader("Drill into a gap")

selected_req = st.selectbox("Requirement", filtered["requirement_id"].tolist())
row = filtered[filtered["requirement_id"] == selected_req].iloc[0]
req_row = requirements[requirements["requirement_id"] == selected_req].iloc[0]

col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown(f"**{row['requirement_id']} — {row['requirement']}**")
    st.write(req_row["description"])
    st.markdown(f"Priority: **{row['priority']}** · Status: {status_badge(row['status'])}")
    st.markdown(f"Best candidate: **{row['best_match_id']}** ({row['best_match_title']}) — score **{row['similarity']:.2f}**")
    st.info(row["recommended_action"])

with col2:
    st.markdown("**Suggested test ideas**")
    st.caption("Deterministic, rule-based recommendations — not automatically generated official test cases.")
    ideas = generate_suggestions(req_row["description"])
    for idea in ideas:
        st.markdown(f"- {idea}")

st.divider()
st.download_button(
    "⬇️ Download coverage gaps (CSV)",
    data=dataframe_to_csv_bytes(gaps),
    file_name="coverage_gaps.csv",
    mime="text/csv",
)
