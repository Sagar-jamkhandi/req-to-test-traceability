import streamlit as st

from src.state import init_session_state
from src.suggestions import generate_suggestions
from src.ui_components import guard_no_analysis, status_badge

init_session_state()

st.title("💡 Test Suggestions")
st.caption(
    "Rule-based test ideas for requirements with weak or missing coverage. "
    "These are recommendations to consider, not automatically generated "
    "official test cases — they come from a deterministic keyword lookup, "
    "not a generative model."
)

if guard_no_analysis():
    st.stop()

gaps = st.session_state["coverage_gaps"]
requirements = st.session_state["requirements_clean"]

if gaps.empty:
    st.success("No weak or missing requirements at the current thresholds.")
    st.stop()

for _, row in gaps.iterrows():
    req_row = requirements[requirements["requirement_id"] == row["requirement_id"]].iloc[0]
    with st.expander(f"{row['requirement_id']} — {row['requirement']}  ·  {status_badge(row['status'])}"):
        st.write(req_row["description"])
        st.markdown(f"Priority: **{row['priority']}**  ·  Best match score: **{row['similarity']:.2f}**")
        st.markdown("**Suggested test ideas:**")
        ideas = generate_suggestions(req_row["description"])
        for idea in ideas:
            st.markdown(f"- {idea}")
