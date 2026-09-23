import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src` importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_requirements_df():
    return pd.DataFrame(
        {
            "requirement_id": ["REQ-001", "REQ-002", "REQ-003"],
            "title": [
                "User login",
                "Password reset",
                "Account lockout",
            ],
            "description": [
                "Users shall be able to log in with a valid email and password.",
                "Users shall be able to reset their password using their registered email.",
                "The system shall lock the account after five failed login attempts.",
            ],
            "priority": ["High", "High", "Medium"],
        }
    )


@pytest.fixture
def sample_test_cases_df():
    return pd.DataFrame(
        {
            "test_case_id": ["TC-001", "TC-002", "TC-003", "TC-004"],
            "title": [
                "Verify successful login",
                "Verify password reset email",
                "Verify account lockout after failures",
                "Verify terms checkbox is shown",
            ],
            "description": [
                "Log in with a valid registered email and password and verify success.",
                "Request a password reset and verify a reset email is sent.",
                "Attempt five failed logins and verify the account becomes locked.",
                "Verify the terms and conditions checkbox appears on the signup page.",
            ],
            "priority": ["High", "High", "Medium", "Low"],
        }
    )
