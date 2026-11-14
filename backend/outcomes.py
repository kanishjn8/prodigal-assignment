import re
from typing import Any

FIELD_LABELS = {
    "email": r"\be-?mail\b",
    "cgpa": r"\b(?:cgpa|gpa|grades?)\b",
    "college": r"\b(?:college|university)\b",
    "degree": r"\bdegree\b",
    "branch": r"\b(?:branch|speciali[sz]ation)\b",
    "graduation_year": r"\bgraduation\b",
    "preferred_location": r"\blocation\b",
    "available_from": r"\b(?:start date|availability|available to start|date when|yyyy-mm-dd)\b",
    "internship_duration_months": r"\b(?:duration|(?:how many|number of) months)\b",
    "full_name": r"\bname\b",
}


def requested_exit(message: str) -> dict[str, Any] | None:
    """Recognize clear stop requests; leave other interpretation to the LLM."""
    text = message.casefold().replace("’", "'")
    human = (
        r"\b(?:wrong (?:person|number)|not me|someone else's|not my (?:record|application))\b|"
        r"\b(?:speak|talk|connect|transfer|want|need).{0,30}\b(?:human|person|supervisor)\b|"
        r"\b(?:i'm|i am) (?:angry|upset|frustrated)\b|"
        r"\bthis is (?:annoying|frustrating|upsetting)\b|"
        r"\bstop (?:asking|contacting|messaging)\b|"
        r"\b(?:record|college|degree|name|email|cgpa).{0,20}"
        r"\b(?:is wrong|is incorrect|isn't mine)\b"
    )
    if re.search(human, text):
        return {
            "collection_status": "needs_human",
            "notes": "Candidate request: " + message,
            "refused_fields": [],
        }
    refused = set()
    for clause in re.split(r"[.;\n]|\bbut\b", text):
        if re.search(
            r"\b(?:won't|will not|refuse|don't want to|do not want to|prefer not to)\b", clause
        ):
            refused.update(
                key for key, pattern in FIELD_LABELS.items() if re.search(pattern, clause)
            )
    if refused:
        return {
            "collection_status": "refused",
            "notes": "Candidate refusal: " + message,
            "refused_fields": sorted(refused),
        }
    if re.search(
        r"\b(?:continue|resume|talk|chat|finish|do this|try again)\b.{0,40}"
        r"\b(?:later|tomorrow|next week|tonight|another time|at \d)\b|"
        r"\b(?:i'm busy|i am busy|not now|remind me later)\b",
        text,
    ):
        return {
            "collection_status": "in_progress",
            "notes": "Continue later: " + message,
            "refused_fields": [],
        }
    return None


def outcome_notes(previous: str, context: str) -> str:
    if len(context) > 3500:
        context = "[Context shortened]\n" + context[-3450:]
    if not previous:
        return context
    remaining = 4000 - len(context) - 1
    if len(previous) > remaining:
        marker = "[Earlier notes shortened]\n"
        previous = marker + previous[-max(0, remaining - len(marker)) :]
    return previous + "\n" + context
