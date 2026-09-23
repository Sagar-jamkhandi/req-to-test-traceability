import plotly.graph_objects as go
import streamlit as st

from src.matching_engine import explain_match, top_matches_for_requirement
from src.state import init_session_state
from src.ui_components import guard_no_analysis, status_badge
from src.utils import STATUS_COLORS

init_session_state()

st.title("🔍 Matching Analysis")
st.caption(
    "Select a requirement to see its top candidate test cases, the "
    "TF-IDF vs. semantic score breakdown, and a plain-language explanation "
    "of each match."
)

if guard_no_analysis():
    st.stop()

requirements = st.session_state["requirements_clean"]
test_cases = st.session_state["test_cases_clean"]
match_table = st.session_state["match_table"]
top_n = st.session_state.get("top_n", 5)

req_ids = requirements["requirement_id"].tolist()
labels = {
    r: f"{r} — {requirements[requirements['requirement_id'] == r]['title'].iloc[0]}"
    for r in req_ids
}
selected_id = st.selectbox("Requirement", req_ids, format_func=lambda r: labels[r])

req_row = requirements[requirements["requirement_id"] == selected_id].iloc[0]
st.markdown(f"### {req_row['requirement_id']} — {req_row['title']}")
st.write(req_row["description"])

n_show = st.slider("Number of candidate test cases to show", 1, 10, top_n)
top_matches = top_matches_for_requirement(match_table, selected_id, top_n=n_show)

st.subheader("Top matching test cases")

table_display = top_matches.copy()
table_display["status_label"] = table_display["status"].apply(status_badge)
st.dataframe(
    table_display[
        ["test_case_id", "tfidf_score", "semantic_score", "final_score", "status_label"]
    ].rename(
        columns={
            "test_case_id": "Test Case ID",
            "tfidf_score": "TF-IDF",
            "semantic_score": "Semantic",
            "final_score": "Final",
            "status_label": "Status",
        }
    ).style.format({"TF-IDF": "{:.2f}", "Semantic": "{:.2f}", "Final": "{:.2f}"}),
    width='stretch',
    hide_index=True,
)

# Small comparative bar chart: TF-IDF vs Semantic per candidate
fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="TF-IDF",
        x=top_matches["test_case_id"],
        y=top_matches["tfidf_score"],
        marker_color="#8FA8D6",
    )
)
fig.add_trace(
    go.Bar(
        name="Semantic",
        x=top_matches["test_case_id"],
        y=top_matches["semantic_score"],
        marker_color="#2E5AAC",
    )
)
fig.update_layout(
    barmode="group",
    height=300,
    margin=dict(t=20, b=10, l=10, r=10),
    yaxis_title="Score",
    xaxis_title="Candidate test case",
)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Match details & explanation")

for _, row in top_matches.iterrows():
    tc_row = test_cases[test_cases["test_case_id"] == row["test_case_id"]].iloc[0]
    with st.expander(
        f"{row['test_case_id']} — {tc_row['title']}  ·  score {row['final_score']:.2f}  ·  {status_badge(row['status'])}"
    ):
        st.write(tc_row["description"])
        explanation = explain_match(
            requirement_text=req_row["combined_text_raw"],
            test_case_text=tc_row["combined_text_raw"],
            tfidf_score=row["tfidf_score"],
            semantic_score=row["semantic_score"],
            final_score=row["final_score"],
            status=row["status"],
        )
        st.info(explanation)
