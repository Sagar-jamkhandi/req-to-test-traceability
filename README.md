# Requirement → Test Traceability Builder

An AI-assisted QA tool that maps requirements/user stories to test cases,
classifies coverage as **Strong / Partial / Missing**, and helps a QA
engineer quickly see where test planning attention is needed.

Built for a "Requirement to Test Traceability Builder" take-home
challenge (target scope: ~15 requirements / ~12 test cases,
build budget).

---

## 1. Project overview

Given a set of requirements and a set of test cases, this app:

- Matches every requirement against every test case using two
  complementary NLP techniques (TF-IDF and semantic embeddings).
- Classifies each requirement's best-matching test case as Strong,
  Partial, or Missing coverage, using configurable thresholds.
- Builds a many-to-many traceability matrix.
- Surfaces coverage gaps, sorted by priority and score.
- Generates deterministic, rule-based test-idea suggestions for weak or
  missing requirements.
- Explains *why* a requirement and test case were matched, in plain
  language, with a TF-IDF vs. semantic score breakdown.
- Exports results to CSV/Excel.

## 2. Problem statement

Manually maintaining a requirement-to-test traceability matrix is slow
and error-prone, especially as a project grows. QA teams need a fast way
to answer: *"Which requirements don't have adequate test coverage yet,
and what should we test next?"*

## 3. Objectives

1. Accurate requirement-to-test-case mapping.
2. Clear, configurable coverage status per requirement.
3. Easy identification of missing/weak coverage.
4. Useful, actionable test-planning suggestions.
5. Explainability — every match can be justified in plain language.
6. A professional, focused UI suitable for a live demo.

## 4. Architecture

```
Streamlit UI (app.py registers every page via st.navigation() + pages/)
        │
        ▼
src/  (pure-Python core library, independently testable)
  ├─ data_loader.py     – CSV/XLSX ingestion, flexible column matching
  ├─ validation.py      – friendly validation, never crashes on bad input
  ├─ preprocessing.py   – text cleaning, de-duplication
  ├─ tfidf_matcher.py   – TF-IDF + cosine similarity baseline
  ├─ semantic_matcher.py– Sentence-Transformer embeddings + cosine similarity
  ├─ matching_engine.py – weighted score combination + explainability
  ├─ coverage.py        – classification, traceability matrix, gap analysis
  ├─ suggestions.py     – deterministic, rule-based test idea generator
  ├─ exporters.py       – CSV / Excel export
  ├─ pipeline.py        – orchestrates the full run, with Streamlit caching
  ├─ state.py           – centralized st.session_state management
  └─ ui_components.py   – shared rendering helpers
```

The `src/` package has no Streamlit-specific logic baked into its core
functions (aside from `pipeline.py`/`state.py`), so it is directly unit
testable with `pytest` and could be reused outside Streamlit.

`app.py` explicitly registers every page with `st.Page`/`st.navigation()`
rather than relying on Streamlit's implicit `pages/`-folder auto-discovery.
This is the more robust, deployment-agnostic pattern recommended by
Streamlit, and avoids sidebar/navigation issues some local setups can hit
with folder auto-discovery. Because of this, individual files under
`pages/` do **not** call `st.set_page_config()` themselves — it's called
exactly once, in `app.py`.

## 5. Technology stack

- **Python 3.11+**, **Streamlit** (UI)
- **Pandas / NumPy** (data handling)
- **scikit-learn** (TF-IDF + cosine similarity)
- **Sentence-Transformers** (`all-MiniLM-L6-v2` semantic embeddings)
- **Plotly** (charts)
- **Pytest** (testing)
- **openpyxl** (Excel export)

No React, FastAPI, databases, message queues, or cloud infrastructure —
intentionally kept to a single, focused Streamlit application.

## 6. Matching methodology

For every (requirement, test case) pair:

```
final_score = tfidf_weight * tfidf_score + semantic_weight * semantic_score
```

- **TF-IDF score**: cosine similarity between TF-IDF vectors, fit jointly
  across all requirements + test cases so both sides share a vocabulary.
- **Semantic score**: cosine similarity between `all-MiniLM-L6-v2`
  sentence embeddings. If the model can't be loaded (e.g. no network
  access), the app **automatically falls back to the TF-IDF score** for
  this component and shows a warning banner — it never crashes.
- Weights default to **30% TF-IDF / 70% semantic** and are configurable
  in the UI. Weights are normalized to sum to 1.0 automatically.

### TF-IDF explanation

TF-IDF ("term frequency – inverse document frequency") scores how
important a word is to a document relative to the whole corpus. Cosine
similarity between two TF-IDF vectors measures lexical/keyword overlap.
It's fast, fully deterministic, and a solid baseline — but it misses
matches where the *wording* differs even though the *meaning* is the
same (e.g. "log in" vs. "authenticate").

### Semantic similarity explanation

Sentence-Transformer embeddings map each requirement/test case to a
dense vector that captures meaning, not just words. Cosine similarity
between embeddings can catch paraphrased matches that TF-IDF misses, at
the cost of being less transparent about *why* two texts matched (which
is why every match also gets a plain-language explanation and shared-term
highlighting).

## 7. Coverage methodology

Each requirement's status is based on its **single best-matching test
case** (highest `final_score`):

| Status | Condition |
|---|---|
| Strong | best score ≥ Strong threshold (default 0.75) |
| Partial | Partial threshold ≤ best score < Strong threshold (default 0.50–0.75) |
| Missing / Weak | best score < Partial threshold |

Overall coverage %:

```
coverage % = (requirements "adequately covered" / total requirements) * 100
```

"Adequately covered" = Strong + Partial by default; can be restricted to
Strong-only via a checkbox on the Data Upload page.

## 8. Assumptions

See the in-app **Assumptions & Methodology** page for the full, live
list (thresholds shown there reflect your current session's settings).
Summary:

- Matching compares requirement **title + description** against test-case
  **title + description** only.
- Similarity indicates textual/semantic relevance, **not proof** of
  functional coverage.
- Thresholds and weights are configurable heuristics, not universal
  standards — calibrate them for your project.
- One requirement can map to multiple test cases and vice versa.
- "Missing" means no sufficiently relevant candidate was found among the
  *uploaded* test cases, not that no such test could exist elsewhere.
- Test-idea suggestions come from a fixed keyword lookup table, not a
  generative model.
- Human QA review is still required before treating any result as final.

## 9. Project structure

```
req-to-test-traceability/
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Data_Upload.py
│   ├── 3_Requirements.py
│   ├── 4_Test_Cases.py
│   ├── 5_Traceability.py
│   ├── 6_Coverage_Gaps.py
│   ├── 7_Test_Suggestions.py
│   ├── 8_Matching_Analysis.py
│   └── 9_Assumptions.py
├── src/
│   ├── data_loader.py, validation.py, preprocessing.py
│   ├── tfidf_matcher.py, semantic_matcher.py, matching_engine.py
│   ├── coverage.py, suggestions.py, exporters.py
│   ├── pipeline.py, state.py, ui_components.py, utils.py
├── data/
│   ├── sample_requirements.csv (15 rows)
│   ├── sample_test_cases.csv (20 rows)
│   └── templates/ (blank CSV templates)
├── tests/ (43 pytest tests)
├── .streamlit/config.toml
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 10. Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run will download the `all-MiniLM-L6-v2` model (~80 MB) from
Hugging Face; this requires internet access once. If it can't download,
the app automatically falls back to TF-IDF-only scoring and tells you so
in the UI.

## 11. Running the application

```bash
streamlit run app.py
```

Then, in the app:
1. Go to **Data Upload** → click **Load sample data** (or upload your own
   CSV/XLSX files).
2. Review the validation preview.
3. Adjust weights/thresholds if desired (optional — sensible defaults are
   pre-filled).
4. Click **Run Traceability Analysis**.
5. Explore **Dashboard**, **Traceability Matrix**, **Coverage Gaps**,
   **Test Suggestions**, and **Matching Analysis**.

## 12. Running tests

```bash
pytest
```

43 tests cover validation, preprocessing, TF-IDF scoring, the combined
matching engine (including the semantic-model fallback path), coverage
classification/aggregation, and the suggestion rules.

## 13. Sample data

`data/sample_requirements.csv` / `data/sample_test_cases.csv` model a
"User Authentication and Account Management System" feature (15
requirements, 20 test cases), designed to demonstrate:

- Strong coverage (clear, well-matched tests)
- Partial coverage (related but incomplete tests)
- Missing coverage (2FA, account recovery via security questions, and
  audit logging have no matching test case)
- Multiple test cases covering one requirement (e.g. registration,
  invalid login, password reset, email verification, account lockout)
- A test case that is lexically similar but semantically different
  (a UI "password masking" test vs. the password-complexity requirement)
- An irrelevant test case (terms & conditions checkbox) with no strong
  requirement match

## 14. Limitations

- Only title + description are analyzed — test steps, preconditions, and
  attachments are not read.
- Small datasets make TF-IDF vocabulary sparse, which can make lexical
  scores look more volatile than on a larger corpus.
- Semantic similarity can produce false positives/negatives, especially
  on short or ambiguous text.
- This tool supports QA judgment; it does not replace it.

## 15. Future improvements

- Persist analysis runs and compare coverage over time.
- Allow editing requirements/test cases in-app instead of only via file
  upload.
- Support additional embedding models and let users A/B compare them
  visually.
- Add per-requirement manual override of coverage status with an audit
  trail.
- Support test steps / expected results as additional matched fields.
