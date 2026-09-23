import streamlit as st

from src.state import init_session_state
from src.utils import DEFAULT_PARTIAL_THRESHOLD, DEFAULT_STRONG_THRESHOLD

init_session_state()

st.title("📖 Assumptions & Methodology")
st.caption("Read this before treating any score in this app as a final QA decision.")

st.markdown(
    """
### What this tool is — and isn't

This is a **decision-support tool** for QA planning. It estimates how
*textually and semantically relevant* a test case is to a requirement. It
does **not** prove that a requirement is functionally covered, and it
cannot verify that a test actually exercises the described behavior
correctly. Treat every score as a starting point for human review, not a
final answer.

### Matching methodology

1. Each requirement and test case is represented as **title + description**
   combined into a single text ("combined text").
2. Text is lightly cleaned (lowercased, whitespace/punctuation normalized).
   Stemming, stop-word removal, and lemmatization are deliberately **not**
   applied, to avoid destroying short, domain-term-dense QA phrasing.
3. Two independent relevance signals are computed for every
   (requirement, test case) pair:
   - **TF-IDF + cosine similarity** — a lexical/keyword-overlap baseline,
     fit jointly across all requirements and test cases in the data set.
   - **Semantic similarity** — cosine similarity between
     Sentence-Transformer embeddings (`all-MiniLM-L6-v2` by default), which
     can recognize related meaning even with different wording. If the
     model cannot be loaded (e.g. no network access), the app **falls back
     to TF-IDF automatically** and displays a warning — it never silently
     fails.
4. The two scores are combined into a single **final score** using
   user-configurable weights (default: 30% TF-IDF, 70% semantic). These
   weights are heuristics recommended as a reasonable starting point, not
   a universally validated ratio — recalibrate them for your own domain
   and data.

### Coverage classification

Each requirement's coverage status is based on its **single best-matching
test case** (the highest `final_score` among all test cases), not an
average across candidates. Configurable thresholds classify that best
score as:
"""
)

strong_t = st.session_state.get("strong_threshold", DEFAULT_STRONG_THRESHOLD)
partial_t = st.session_state.get("partial_threshold", DEFAULT_PARTIAL_THRESHOLD)

st.markdown(
    f"""
- **Strong** — best score ≥ **{strong_t:.2f}**
- **Partial** — best score ≥ **{partial_t:.2f}** and < {strong_t:.2f}
- **Missing / Weak** — best score < **{partial_t:.2f}**

These are the thresholds currently configured on the Data Upload page
(defaults: Strong ≥ 0.75, Partial ≥ 0.50). They are heuristic cut-points,
**not** industry or regulatory QA standards, and should be calibrated
against your own historical project data if you rely on this for audit or
compliance purposes.

### Overall coverage percentage

```
coverage % = (requirements considered "adequately covered" / total requirements) * 100
```

"Adequately covered" can mean **Strong only**, or **Strong + Partial** —
this is a configurable checkbox on the Data Upload page. The app defaults
to counting Strong + Partial as adequate, since Partial coverage still
represents a relevant, human-reviewable candidate test.

### Many-to-many traceability

- One requirement may have **multiple relevant test cases** — all
  candidates above a small relevance floor are shown in the Traceability
  Matrix (up to a configurable top-N per requirement).
- One test case may appear against **multiple requirements** if it is
  relevant to more than one.
- "Missing coverage" means **no sufficiently relevant candidate test case
  was identified by the matching pipeline** at the current thresholds — it
  does not necessarily mean no such test exists in your broader test
  suite, only that nothing in the uploaded test-case file scored highly
  enough against this requirement.

### Test idea suggestions

Suggestions on the Coverage Gaps and Test Suggestions pages are produced
by a **fixed, deterministic keyword → test-idea lookup table** (e.g. a
requirement mentioning "login" surfaces valid/invalid-credential and
lockout test ideas). No generative AI model is used for this feature —
the trade-off is less flexibility in exchange for full explainability,
reproducibility, and zero risk of fabricated content. Suggestions are
recommendations to consider, not automatically generated official test
cases.

### Known limitations

- Semantic similarity can produce **false positives** (unrelated text that
  happens to embed closely) and **false negatives** (related text embedded
  far apart), especially for short or ambiguous phrasing.
- The matchers only see **title + description** — test steps, expected
  results, preconditions, or linked attachments are not analyzed if they
  live outside those two fields.
- Small data sets (as in this challenge's ~15/~20 scale) make TF-IDF
  vocabulary sparse; scores can look more volatile than they would on a
  larger corpus.
- **Human QA review is still required** before treating any result here as
  confirmed coverage or a confirmed gap.
"""
)
