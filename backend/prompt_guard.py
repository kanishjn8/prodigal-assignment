import re
import unicodedata

ROLE_MARKER = re.compile(
    r"<\|[^<>\s]{1,64}\|>|\[/?INST\]|<\/?(?:system|assistant|developer|tool|user)\s*>",
    re.IGNORECASE,
)
OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|disable)\b[^.!?\n]{0,64}"
    r"\b(?:instructions?|rules?|polic(?:y|ies)|validation|guards?|confirmation)\b"
    r"|\b(?:print|show|reveal|dump|expose|repeat)\b[^.!?\n]{0,64}"
    r"\b(?:system prompt|developer message|api key|bearer token|credentials)\b"
    r"|\b(?:call|invoke|execute|run)\b[^.!?\n]{0,40}"
    r"\b(?:inspect_candidate_state|validate_candidate_information|"
    r"save_candidate_information|finish_collection)\b",
    re.IGNORECASE,
)


def contains_prompt_override(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Cf")
    return bool(ROLE_MARKER.search(normalized) or OVERRIDE.search(normalized))
