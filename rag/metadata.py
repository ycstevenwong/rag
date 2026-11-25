"""Fixed lists for document labels.

Version rules match on these labels, so a misspelled one would apply the wrong rule
with no warning. New labels have to come from the configured lists.
"""
from __future__ import annotations


def canonical_label(value, allowed: list[str], field: str) -> str:
    """Return `value` in its listed spelling, ignoring case and spacing. Raises
    ValueError if it isn't listed. An empty list means this label isn't restricted."""
    text = value.strip() if isinstance(value, str) else ""
    if not allowed:
        return text
    for option in allowed:
        if option.casefold() == text.casefold():
            return option
    raise ValueError(f"Choose a {field} from: {', '.join(allowed)}.")
