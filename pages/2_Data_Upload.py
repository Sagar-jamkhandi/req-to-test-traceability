import streamlit as st

from src.data_loader import (
    DataLoadError,
    load_requirements,
    load_sample_requirements,
    load_sample_test_cases,
    load_test_cases,
    requirement_template_csv,
    test_case_template_csv,
)
from src.pipeline import run_full_analysis
from src.preprocessing import preprocess_requirements, preprocess_test_cases
from src.state import init_session_state, reset_analysis_state
from src.ui_components import safe_page_link
from src.utils import (
    DEFAULT_PARTIAL_THRESHOLD,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_TFIDF_WEIGHT,
    DEFAULT_TOP_N,
)
from src.validation import (
    validate_requirements,
    validate_test_cases,
    validate_thresholds,
    validate_weights,
)

init_session_state()

st.title("📤 Data Upload")
st.caption("Load requirements and test cases, then run the traceability analysis.")

# --------------------------------------------------------------------
# Step 1: choose data source
# --------------------------------------------------------------------
st.subheader("1. Choose a data source")

source = st.radio(
    "Data source",
    options=["Use bundled sample data", "Upload my own files"],
    horizontal=True,
    label_visibility="collapsed",
)

req_df_raw, tc_df_raw = None, None
req_errors, tc_errors = [], []

if source == "Use bundled sample data":
    st.info(
        "Sample data: a 'User Authentication and Account Management System' "
        "feature with 15 requirements and 20 test cases, including "
        "intentionally strong, partial, and missing coverage scenarios."
    )
    if st.button("Load sample data", type="primary"):
        req_df_raw = load_sample_requirements()
        tc_df_raw = load_sample_test_cases()
        st.session_state["data_source"] = "sample"
        st.session_state["requirements_raw"] = req_df_raw
        st.session_state["test_cases_raw"] = tc_df_raw
        reset_analysis_state()

else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Requirements file** (CSV or XLSX)")
        req_file = st.file_uploader(
            "Requirements file", type=["csv", "xlsx"], label_visibility="collapsed"
        )
        st.download_button(
            "⬇️ Download requirements template",
            data=requirement_template_csv(),
            file_name="requirements_template.csv",
            mime="text/csv",
        )
    with col_b:
        st.markdown("**Test cases file** (CSV or XLSX)")
        tc_file = st.file_uploader(
            "Test cases file", type=["csv", "xlsx"], label_visibility="collapsed"
        )
        st.download_button(
            "⬇️ Download test case template",
            data=test_case_template_csv(),
            file_name="test_cases_template.csv",
            mime="text/csv",
        )

    expected_req_cols = "requirement_id, title, description, priority (optional)"
    expected_tc_cols = "test_case_id, title, description, priority (optional)"
    st.caption(
        f"Expected requirement columns: {expected_req_cols}. Column names are "
        f"matched flexibly (e.g. 'Req ID', 'Requirement Id' are both recognized)."
    )

    if req_file is not None:
        try:
            req_df_raw = load_requirements(req_file, filename=req_file.name)
        except DataLoadError as exc:
            st.error(f"Requirements file: {exc}")

    if tc_file is not None:
        try:
            tc_df_raw = load_test_cases(tc_file, filename=tc_file.name)
        except DataLoadError as exc:
            st.error(f"Test cases file: {exc}")

    if req_df_raw is not None and tc_df_raw is not None:
        if st.button("Use these files", type="primary"):
            st.session_state["data_source"] = "upload"
            st.session_state["requirements_raw"] = req_df_raw
            st.session_state["test_cases_raw"] = tc_df_raw
            reset_analysis_state()

# --------------------------------------------------------------------
# Step 2: validate + preview
# --------------------------------------------------------------------
if st.session_state.get("requirements_raw") is not None:
    st.subheader("2. Validate and preview")

    req_result = validate_requirements(st.session_state["requirements_raw"])
    tc_result = validate_test_cases(st.session_state["test_cases_raw"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Requirements** ({len(st.session_state['requirements_raw'])} rows)")
        if req_result.errors:
            for e in req_result.errors:
                st.error(e)
        for w in req_result.warnings:
            st.warning(w)
        if req_result.is_valid:
            st.dataframe(
                st.session_state["requirements_raw"].head(10), width='stretch'
            )

    with col2:
        st.markdown(f"**Test cases** ({len(st.session_state['test_cases_raw'])} rows)")
        if tc_result.errors:
            for e in tc_result.errors:
                st.error(e)
        for w in tc_result.warnings:
            st.warning(w)
        if tc_result.is_valid:
            st.dataframe(
                st.session_state["test_cases_raw"].head(10), width='stretch'
            )

    data_is_valid = req_result.is_valid and tc_result.is_valid

    if data_is_valid:
        st.session_state["requirements_clean"] = preprocess_requirements(
            st.session_state["requirements_raw"]
        )
        st.session_state["test_cases_clean"] = preprocess_test_cases(
            st.session_state["test_cases_raw"]
        )
        st.session_state["requirements_warnings"] = req_result.warnings
        st.session_state["test_cases_warnings"] = tc_result.warnings

        n_req = len(st.session_state["requirements_clean"])
        n_tc = len(st.session_state["test_cases_clean"])
        st.success(
            f"Data is valid: {n_req} requirement(s) and {n_tc} test case(s) "
            f"ready for analysis after cleaning."
        )

        # ----------------------------------------------------------------
        # Step 3: configure matching
        # ----------------------------------------------------------------
        st.subheader("3. Configure matching")
        st.caption(
            "These are heuristic defaults, not universal QA standards — "
            "adjust them to fit your project and re-run the analysis."
        )

        with st.expander("Matching weights & thresholds", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Score weighting**")
                tfidf_weight = st.slider(
                    "TF-IDF weight", 0.0, 1.0, DEFAULT_TFIDF_WEIGHT, 0.05
                )
                semantic_weight = st.slider(
                    "Semantic weight", 0.0, 1.0, DEFAULT_SEMANTIC_WEIGHT, 0.05
                )
                st.caption(
                    "`final_score = tfidf_weight * tfidf_score + semantic_weight * semantic_score` "
                    "(weights are normalized to sum to 1.0)."
                )
            with c2:
                st.markdown("**Coverage thresholds**")
                strong_threshold = st.slider(
                    "Strong coverage threshold (≥)", 0.0, 1.0, DEFAULT_STRONG_THRESHOLD, 0.05
                )
                partial_threshold = st.slider(
                    "Partial coverage threshold (≥)", 0.0, 1.0, DEFAULT_PARTIAL_THRESHOLD, 0.05
                )
                top_n = st.slider("Candidate test cases shown per requirement", 1, 10, DEFAULT_TOP_N)

            adequate_includes_partial = st.checkbox(
                "Count 'Partial' as adequately covered in the overall coverage %",
                value=True,
                help="If unchecked, only 'Strong' coverage counts toward the headline coverage percentage.",
            )

        threshold_check = validate_thresholds(strong_threshold, partial_threshold)
        weight_check = validate_weights(tfidf_weight, semantic_weight)

        for e in threshold_check.errors:
            st.error(e)
        for w in weight_check.warnings:
            st.info(w)

        config_is_valid = threshold_check.is_valid

        # ----------------------------------------------------------------
        # Step 4: run analysis
        # ----------------------------------------------------------------
        st.subheader("4. Run analysis")

        run_disabled = not config_is_valid
        if st.button("🚀 Run Traceability Analysis", type="primary", disabled=run_disabled):
            with st.spinner("Matching requirements to test cases..."):
                results = run_full_analysis(
                    st.session_state["requirements_clean"],
                    st.session_state["test_cases_clean"],
                    tfidf_weight,
                    semantic_weight,
                    strong_threshold,
                    partial_threshold,
                    top_n,
                    adequate_includes_partial,
                )
            st.session_state.update(results)
            st.session_state["tfidf_weight"] = tfidf_weight
            st.session_state["semantic_weight"] = semantic_weight
            st.session_state["strong_threshold"] = strong_threshold
            st.session_state["partial_threshold"] = partial_threshold
            st.session_state["top_n"] = top_n
            st.session_state["adequate_includes_partial"] = adequate_includes_partial
            st.session_state["analysis_has_run"] = True

            if results["semantic_model_loaded"]:
                st.success(
                    "Analysis complete. Semantic model (all-MiniLM-L6-v2) loaded successfully."
                )
            else:
                st.warning(
                    "Analysis complete, but the semantic model could not be loaded "
                    f"({results['semantic_load_error']}). The app automatically fell "
                    "back to TF-IDF-only scoring for the semantic component."
                )
            safe_page_link("pages/1_Dashboard.py", "➡️ View the Dashboard", "📊")
    else:
        st.error(
            "Please fix the errors above before running the analysis. If your "
            "column names don't match, check the expected format or use the "
            "provided templates."
        )
