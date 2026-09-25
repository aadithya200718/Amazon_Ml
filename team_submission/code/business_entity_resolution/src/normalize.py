"""
Normalization module for business names, addresses, and country codes.
Amazon ML Challenge 2026.
"""

import re
import unicodedata

# Multi-country legal suffixes (US, India, France)
SUFFIX_REGEX = (
    r"\b(inc|incorporated|corp|corporation|llc|llp|ltd|limited|co|company|"
    r"pvt|private|services|enterprises|solutions|group|holdings|industries|"
    r"international|center|centre|club|store|shop|care|management|"
    r"sarl|sasu|sas|eurl|sci|sa|snc|gie)\b"
)

# Multi-country address stop words & generic street markers
ADDR_STOP_REGEX = (
    r"\b(the|and|of|in|for|on|at|to|a|an|is|by|with|"
    r"road|rd|street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|way|court|ct|"
    r"circle|cir|suite|ste|apt|unit|floor|fl|building|bldg|north|south|east|west|"
    r"box|po|pobox|near|opp|behind|none|null|usa|us|india|france|fr|saint|ste|"
    r"rue|bd|allee|allée|chemin|place|cours|route|rte)\b"
)

# Canonical country mapping (unrecognized labels pass through safely)
COUNTRY_SYNONYMS = {
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "us": "US",
    "india": "India",
    "in": "India",
    "france": "France",
    "fr": "France"
}


def canonical_country(raw: str) -> str:
    """Normalize country string safely, passing through unseen countries."""
    if not raw or not isinstance(raw, str):
        return ""
    key = raw.strip().lower()
    return COUNTRY_SYNONYMS.get(key, raw.strip())


def normalize_text(text: str) -> str:
    """Basic unicode normalization and lowercasing."""
    if not text or not isinstance(text, str) or text.lower() in ("none", "null"):
        return ""
    s = unicodedata.normalize("NFKD", text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
