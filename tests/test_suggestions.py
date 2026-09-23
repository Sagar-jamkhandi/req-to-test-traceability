from src.suggestions import generate_suggestions


def test_login_keyword_triggers_login_suggestions():
    suggestions = generate_suggestions("Users shall be able to log in with valid credentials")
    assert any("login" in s.lower() for s in suggestions)


def test_password_keyword_triggers_password_suggestions():
    suggestions = generate_suggestions("Passwords must meet complexity requirements")
    assert any("password" in s.lower() for s in suggestions)


def test_upload_keyword_triggers_upload_suggestions():
    suggestions = generate_suggestions("Users shall be able to upload a profile picture")
    joined = " ".join(suggestions).lower()
    assert "file" in joined or "upload" in joined


def test_unmatched_text_returns_generic_fallback():
    suggestions = generate_suggestions("The system shall render the dashboard in under two seconds")
    assert len(suggestions) > 0


def test_suggestions_are_deduplicated_across_multiple_keywords():
    suggestions = generate_suggestions("Login requires a valid password and email")
    assert len(suggestions) == len(set(suggestions))


def test_max_suggestions_is_respected():
    suggestions = generate_suggestions("login password email session lockout", max_suggestions=3)
    assert len(suggestions) <= 3


def test_empty_text_returns_fallback():
    suggestions = generate_suggestions("")
    assert len(suggestions) > 0
