import numpy as np

from src.tfidf_matcher import TfidfMatcher


def test_identical_text_scores_high():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix(
        ["user can reset password using email"],
        ["user can reset password using email"],
    )
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] > 0.99


def test_unrelated_text_scores_low():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix(
        ["user can reset password using email"],
        ["completely unrelated topic about weather forecasting"],
    )
    assert matrix[0, 0] < 0.2


def test_similar_text_scores_moderate_to_high():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix(
        ["user can reset password using registered email"],
        ["verify password recovery through registered email"],
    )
    assert matrix[0, 0] > 0.2


def test_score_matrix_shape():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix(
        ["req one text", "req two text"],
        ["test one text", "test two text", "test three text"],
    )
    assert matrix.shape == (2, 3)


def test_empty_inputs_return_zero_matrix():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix([], ["some test text"])
    assert matrix.shape == (0, 1)


def test_scores_are_within_bounds():
    matcher = TfidfMatcher()
    matrix = matcher.score_matrix(
        ["alpha beta gamma", "delta epsilon"],
        ["alpha beta", "gamma delta epsilon zeta"],
    )
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
