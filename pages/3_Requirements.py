import streamlit as st

from src.state import init_session_state
from src.ui_components import guard_no_data, status_badge

init_session_state()

st.title("📋 Requirements")

if guard_no_data():
    st.stop()

requirements = st.session_state["requirements_clean"]
warnings = st.session_state.get("requirements_warnings", [])

if warnings:
    with st.expander(f"⚠️ {len(warnings)} data-quality note(s) from validation", expanded=False):
        for w in warnings:
            st.warning(w)

st.caption(f"{len(requirements)} requirement(s) after cleaning and de-duplication.")

# Optional priority filter
priorities = ["All"] + sorted(requirements["priority"].unique().tolist())
selected_priority = st.selectbox("Filter by priority", priorities)

display_df = requirements
if selected_priority != "All":
    display_df = display_df[display_df["priority"] == selected_priority]

if st.session_state.get("analysis_has_run"):
    best_match = st.session_state["best_match_table"][["requirement_id", "status", "final_score"]]
    display_df = display_df.merge(best_match, on="requirement_id", how="left")
    display_df["status"] = display_df["status"].apply(status_badge)
    display_df = display_df.rename(columns={"final_score": "best_match_score"})
    cols = ["requirement_id", "title", "description", "priority", "status", "best_match_score"]
else:
    st.info("Run the analysis on the Data Upload page to see coverage status here.")
    cols = ["requirement_id", "title", "description", "priority"]

st.dataframe(display_df[cols], width='stretch', hide_index=True)
