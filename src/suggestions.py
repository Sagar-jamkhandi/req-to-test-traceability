"""Deterministic, rule/template-based test-idea suggestions for
weak/uncovered requirements.

This deliberately does NOT call any generative model - suggestions come
from a fixed keyword -> test-idea lookup table. This keeps the feature
explainable, reproducible, and free of hallucination risk, at the cost of
being less flexible than an LLM. That trade-off is documented on the
Assumptions page.
"""
from __future__ import annotations

from typing import Dict, List

# Keyword -> suggested test ideas. Checked in insertion order; a
# requirement can match multiple keywords and suggestions are de-duplicated.
KEYWORD_RULES: Dict[str, List[str]] = {
    "login": [
        "Verify login with valid credentials",
        "Verify login is rejected with an invalid password",
        "Verify login is rejected with an invalid/unknown username",
        "Verify login is rejected with empty credentials",
        "Verify behavior after repeated failed login attempts",
    ],
    "password": [
        "Verify a valid password is accepted",
        "Verify an invalid/weak password is rejected",
        "Verify password length/complexity boundary conditions",
        "Verify the password reset workflow end-to-end",
        "Verify password reset with an unregistered email",
    ],
    "registration": [
        "Verify registration with valid, unique details",
        "Verify registration is rejected for a duplicate account",
        "Verify registration validation for required/invalid fields",
        "Verify confirmation email/notification is sent after registration",
    ],
    "email": [
        "Verify email format validation",
        "Verify email verification link works and expires correctly",
        "Verify behavior when the verification email is not received",
        "Verify duplicate email addresses are handled correctly",
    ],
    "session": [
        "Verify session is created after successful authentication",
        "Verify session expires after the configured timeout",
        "Verify session is invalidated on logout",
        "Verify concurrent session handling (if applicable)",
    ],
    "logout": [
        "Verify logout ends the active session",
        "Verify protected pages are inaccessible after logout",
        "Verify logout from multiple devices/sessions (if applicable)",
    ],
    "lockout": [
        "Verify account lockout after the configured number of failed attempts",
        "Verify lockout duration/expiry behavior",
        "Verify a locked account cannot log in even with the correct password",
        "Verify unlock workflow (self-service or admin-assisted)",
    ],
    "role": [
        "Verify each role can access only its permitted features",
        "Verify unauthorized access attempts are blocked and logged",
        "Verify role changes take effect immediately or on next login",
    ],
    "access": [
        "Verify access is granted for permitted roles/users",
        "Verify access is denied for unauthorized roles/users",
        "Verify boundary cases around permission changes",
    ],
    "profile": [
        "Verify profile fields can be updated with valid data",
        "Verify invalid profile data is rejected with a clear error",
        "Verify updated profile data persists after re-login",
    ],
    "upload": [
        "Verify a valid file uploads successfully",
        "Verify unsupported file formats are rejected",
        "Verify the maximum file size boundary",
        "Verify behavior with an empty file",
        "Verify behavior with a corrupted file",
    ],
    "notification": [
        "Verify notifications are sent for the relevant event",
        "Verify notification content is accurate",
        "Verify behavior when notification delivery fails",
    ],
    "search": [
        "Verify search returns expected results for a valid query",
        "Verify search handles an empty/no-match query",
        "Verify search performance with a large result set",
    ],
    "payment": [
        "Verify a successful payment with valid details",
        "Verify a payment is rejected with invalid/expired details",
        "Verify handling of payment gateway timeouts/failures",
    ],
}

GENERIC_FALLBACK_SUGGESTIONS = [
    "Verify the primary (happy-path) scenario described by the requirement",
    "Verify behavior with invalid or unexpected input",
    "Verify behavior at relevant boundary conditions",
    "Verify appropriate error handling/messaging",
]


def generate_suggestions(requirement_text: str, max_suggestions: int = 6) -> List[str]:
    """Return deterministic suggested test ideas for a requirement's text.

    Matching is a simple case-insensitive substring/keyword check against
    ``KEYWORD_RULES``. If no keyword matches, a small generic fallback set
    is returned so the caller always has something actionable to show.
    """
    text = (requirement_text or "").lower()
    # Normalize common two-word phrasing to the single-word keyword form
    # used in KEYWORD_RULES (e.g. "log in" / "logged in" -> "login").
    text = text.replace("log in", "login").replace("logged in", "login").replace("logging in", "login")
    suggestions: List[str] = []
    seen = set()

    for keyword, ideas in KEYWORD_RULES.items():
        if keyword in text:
            for idea in ideas:
                if idea not in seen:
                    suggestions.append(idea)
                    seen.add(idea)

    if not suggestions:
        suggestions = list(GENERIC_FALLBACK_SUGGESTIONS)

    return suggestions[:max_suggestions]
