"""Requirement -> Test Traceability Builder

Main entry point. This file is a router: it registers every page with
``st.navigation`` explicitly (rather than relying on Streamlit's implicit
``pages/`` folder auto-discovery), then hands off control to whichever
page the user selects. This is the more robust, deployment-agnostic
multipage pattern recommended by Streamlit, and avoids sidebar/navigation
issues that folder auto-discovery can hit in some local setups.

``st.set_page_config`` may only be called once per app run, so it lives
here and only here - individual page files under ``pages/`` must not call
it themselves.
"""
import os

# Must be set before numpy, pandas, or torch are imported anywhere in the
# process (Streamlit itself imports numpy/pandas as soon as it's imported
# below). On Windows, several scientific-Python packages each bundle their
# own copy of the Intel OpenMP runtime (libiomp5md.dll); if more than one
# copy ends up loaded in the same process, whichever initializes last can
# fail with a generic "DLL initialization routine failed" error that looks
# like a broken install but isn't. This flag tells OpenMP to tolerate the
# duplicate instead of aborting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from src.state import has_analysis, has_data, init_session_state
from src.ui_components import safe_page_link

st.set_page_config(
    page_title="Requirement -> Test Traceability Builder",
    page_icon="🧭",
    layout="wide",
)

init_session_state()


def render_home() -> None:
    st.title("🧭 Requirement → Test Traceability Builder")
    st.caption("AI-assisted QA coverage analysis — decision support for QA engineers, not a replacement for QA judgment.")

    st.markdown(
        """
This tool maps a set of **requirements / user stories** to a set of
**test cases**, and helps you see, at a glance, which requirements have
strong, partial, or missing test coverage.

It uses two complementary AI/NLP techniques to estimate relevance between
a requirement and a test case:

- **TF-IDF + cosine similarity** — a fast lexical baseline.
- **Sentence-Transformer embeddings + cosine similarity** — a semantic
  approach that can recognize related meaning even when the exact wording
  differs.

Both scores are combined with configurable weights into a single
**relevance score**, which is then classified as **Strong**, **Partial**,
or **Missing** coverage using configurable thresholds.
"""
    )

    st.divider()

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Suggested demo flow")
        st.markdown(
            """
            1. **Data Upload** — load the bundled sample data (or upload your own CSV/XLSX) and run the analysis.
            2. **Dashboard** — see headline coverage metrics and charts.
            3. **Traceability Matrix** — browse the full many-to-many requirement ↔ test mapping.
            4. **Coverage Gaps** — see which requirements need attention first, and why.
            5. **Test Suggestions** — get rule-based test ideas for weak/uncovered requirements.
            6. **Matching Analysis** — drill into a single requirement's candidate tests and see the explanation for each score.
            7. **Assumptions & Methodology** — read the documented assumptions behind every number in this app.
            """
        )

    with col2:
        st.subheader("Current session status")
        if has_data():
            st.success("Requirements and test cases are loaded.")
        else:
            st.info("No data loaded yet. Go to **Data Upload** to get started.")

        if has_analysis():
            st.success("Analysis has been run — results are ready to explore.")
        else:
            st.warning("Analysis has not been run yet.")

        safe_page_link("pages/2_Data_Upload.py", "➡️ Go to Data Upload", "📤")

    st.divider()
    st.caption(
        "This is a decision-support tool. Similarity scores indicate textual/semantic "
        "relevance, not proof of functional coverage. Human QA review is still required."
    )


home_page = st.Page(render_home, title="Home", icon="🧭", default=True)
dashboard_page = st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊")
data_upload_page = st.Page("pages/2_Data_Upload.py", title="Data Upload", icon="📤")
requirements_page = st.Page("pages/3_Requirements.py", title="Requirements", icon="📋")
test_cases_page = st.Page("pages/4_Test_Cases.py", title="Test Cases", icon="🧪")
traceability_page = st.Page("pages/5_Traceability.py", title="Traceability Matrix", icon="🧷")
coverage_gaps_page = st.Page("pages/6_Coverage_Gaps.py", title="Coverage Gaps", icon="🕳️")
test_suggestions_page = st.Page("pages/7_Test_Suggestions.py", title="Test Suggestions", icon="💡")
matching_analysis_page = st.Page("pages/8_Matching_Analysis.py", title="Matching Analysis", icon="🔍")
assumptions_page = st.Page("pages/9_Assumptions.py", title="Assumptions & Methodology", icon="📖")

navigation = st.navigation(
    [
        home_page,
        dashboard_page,
        data_upload_page,
        requirements_page,
        test_cases_page,
        traceability_page,
        coverage_gaps_page,
        test_suggestions_page,
        matching_analysis_page,
        assumptions_page,
    ]
)
navigation.run()
