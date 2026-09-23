import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.state import init_session_state
from src.ui_components import guard_no_analysis, render_kpi_row
from src.utils import STATUS_COLORS, STATUS_ORDER

init_session_state()

st.title("📊 Dashboard")
st.caption("Headline coverage metrics for the current analysis run.")

if guard_no_analysis():
    st.stop()

summary = st.session_state["coverage_summary"]
best_match = st.session_state["best_match_table"]
match_table = st.session_state["match_table"]
requirements = st.session_state["requirements_clean"]
test_cases = st.session_state["test_cases_clean"]

if not st.session_state.get("semantic_model_loaded", True):
    st.warning(
        "Semantic model unavailable during this analysis run — semantic scores "
        "fell back to TF-IDF. Similarity separation between requirements may be "
        "smaller than with the sentence-transformer model."
    )

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
render_kpi_row(
    [
        ("Total Requirements", str(summary["total_requirements"]), None),
        ("Total Test Cases", str(len(test_cases)), None),
        ("Overall Coverage %", f"{summary['coverage_pct']}%", "Percent of requirements adequately covered (see Assumptions page for definition)."),
    ]
)
render_kpi_row(
    [
        ("Strong", str(summary["strong"]), None),
        ("Partial", str(summary["partial"]), None),
        ("Missing / Weak", str(summary["missing"]), None),
    ]
)

st.progress(summary["coverage_pct"] / 100)

st.divider()

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Coverage distribution")
    counts = {
        "Strong": summary["strong"],
        "Partial": summary["partial"],
        "Missing": summary["missing"],
    }
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                marker=dict(colors=[STATUS_COLORS[s] for s in counts.keys()]),
                hole=0.45,
            )
        ]
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Similarity score distribution (best match per requirement)")
    fig2 = px.histogram(
        best_match,
        x="final_score",
        nbins=10,
        color_discrete_sequence=["#2E5AAC"],
    )
    fig2.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        xaxis_title="Final relevance score",
        yaxis_title="Requirements",
    )
    st.plotly_chart(fig2, width='stretch')

col3, col4 = st.columns(2)

with col3:
    st.subheader("Requirements by priority")
    if "priority" in requirements.columns:
        priority_counts = requirements["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]
        fig3 = px.bar(priority_counts, x="priority", y="count", color_discrete_sequence=["#2E5AAC"])
        fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, yaxis_title="Requirements")
        st.plotly_chart(fig3, width='stretch')

with col4:
    st.subheader("Top coverage gaps (lowest score)")
    gaps_preview = best_match.merge(
        requirements[["requirement_id", "title", "priority"]], on="requirement_id"
    ).sort_values("final_score").head(8)
    fig4 = px.bar(
        gaps_preview,
        x="final_score",
        y="title",
        orientation="h",
        color="status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"status": STATUS_ORDER},
    )
    fig4.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        xaxis_title="Best match score",
        yaxis_title="",
    )
    st.plotly_chart(fig4, width='stretch')

st.divider()
st.caption(
    "Definitions: Strong/Partial/Missing are configurable score thresholds applied "
    "to the best-matching test case per requirement. See Assumptions & Methodology for details."
)
