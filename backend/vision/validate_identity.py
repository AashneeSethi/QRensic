"""
QRensic Vision Engine - Deterministic Identity Validation Module
----------------------------------------------------------------
Compares merchant text observed on the poster with payee identity decoded from QR.
CRITICAL PRINCIPLE:
A name mismatch alone MUST NOT be treated as fraud.
Personal-name payees (e.g., small merchant using a proprietor's personal UPI)
are treated as AMBIGUOUS, not fraudulent.
"""

import re
from typing import Any, List, Optional, Set, Tuple, Union

from backend.vision.models import IdentityConsistency, IdentityEvidence


# Legal, corporate, and business entity suffixes to strip during normalization
BUSINESS_SUFFIXES = [
    r"\bpvt\s+ltd\b",
    r"\bprivate\s+limited\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bllc\b",
    r"\binc\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\benterprises\b",
]

# Synthetic testing artifact prefixes
TEST_PAYEE_PREFIXES = [
    r"^qrforensic_test_payee_",
    r"^qrforensic_test_",
    r"^test_payee_",
]

# Common heuristics for personal names (two word tokens without commercial keywords)
COMMERCIAL_KEYWORDS = {
    "cafe", "coffee", "store", "shop", "mart", "restaurant", "hotel", "bakery",
    "electronics", "pharmacy", "medical", "motors", "services", "solutions",
    "station", "bazaar", "jewellers", "tailors", "textiles", "salon", "spa"
}


def normalize_entity_name(name: str) -> str:
    """
    Normalize business and payee names by standardizing case, punctuation,
    synthetic test prefixes, and corporate entity suffixes.
    """
    if not name:
        return ""

    s = name.lower()

    # Strip synthetic test prefixes
    for prefix in TEST_PAYEE_PREFIXES:
        s = re.sub(prefix, "", s)

    # Replace punctuation and underscores with spaces
    s = re.sub(r"[\.,\-_/\\#@!$%^&*():;\"']", " ", s)

    # Remove corporate entity suffixes
    for suffix in BUSINESS_SUFFIXES:
        s = re.sub(suffix, " ", s)

    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_likely_personal_name(name: str) -> bool:
    """
    Check if a name string matches typical personal name structure
    (1 to 3 words, devoid of commercial/retail keywords).
    """
    tokens = [t for t in name.split() if len(t) > 1]
    if not tokens:
        return False
    # If any word is a distinct commercial keyword, it's likely a business
    if any(t in COMMERCIAL_KEYWORDS for t in tokens):
        return False
    # 2 or 3 tokens without business words (e.g. "ramesh kumar")
    return 1 <= len(tokens) <= 3


def validate_identity(
    poster_text: Union[str, List[str]],
    payee_identity: str,
) -> IdentityEvidence:
    """
    Deterministically compare visible merchant poster text against the decoded QR payee.

    Returns:
      IdentityEvidence with status: CONSISTENT, AMBIGUOUS, or CONTRADICTORY.
    """
    norm_payee = normalize_entity_name(payee_identity)

    # Process poster texts
    poster_candidates: List[str] = []
    if isinstance(poster_text, str):
        poster_candidates = [poster_text]
    elif isinstance(poster_text, list):
        poster_candidates = poster_text

    norm_posters = [normalize_entity_name(t) for t in poster_candidates if t]
    norm_posters = [p for p in norm_posters if len(p) > 1]

    if not norm_payee:
        return IdentityEvidence(
            status=IdentityConsistency.AMBIGUOUS,
            normalized_poster_names=norm_posters,
            normalized_payee_name="",
            match_score=0.0,
            match_rationale="No payee identity decoded from QR to evaluate.",
            is_personal_payee=False,
        )

    if not norm_posters:
        personal = is_likely_personal_name(norm_payee)
        return IdentityEvidence(
            status=IdentityConsistency.AMBIGUOUS,
            normalized_poster_names=[],
            normalized_payee_name=norm_payee,
            match_score=0.5,
            match_rationale="No visible merchant text found on poster to compare against payee identity.",
            is_personal_payee=personal,
        )

    personal_payee = is_likely_personal_name(norm_payee)
    best_score = 0.0
    best_rationale = ""
    best_status = IdentityConsistency.AMBIGUOUS

    payee_tokens = set(norm_payee.split())

    for poster in norm_posters:
        poster_tokens = set(poster.split())

        # 1. Exact match after normalization
        if norm_payee == poster:
            return IdentityEvidence(
                status=IdentityConsistency.CONSISTENT,
                normalized_poster_names=norm_posters,
                normalized_payee_name=norm_payee,
                match_score=1.0,
                match_rationale=f"Exact match between normalized payee '{norm_payee}' and poster '{poster}'.",
                is_personal_payee=personal_payee,
            )

        # 2. Substring / Containment (e.g. "nova coffee" in "nova coffee pvt ltd" or vice versa)
        if norm_payee in poster or poster in norm_payee:
            score = 0.95
            if score > best_score:
                best_score = score
                best_status = IdentityConsistency.CONSISTENT
                best_rationale = (
                    f"Direct containment match: '{norm_payee}' aligns with poster brand '{poster}'."
                )
                continue

        # 3. Token Jaccard overlap
        if payee_tokens and poster_tokens:
            intersection = payee_tokens.intersection(poster_tokens)
            union = payee_tokens.union(poster_tokens)
            jaccard = len(intersection) / float(len(union))
            if jaccard >= 0.5:
                score = 0.80 + 0.15 * jaccard
                if score > best_score:
                    best_score = score
                    best_status = IdentityConsistency.CONSISTENT
                    best_rationale = f"Significant token overlap ({intersection}) between payee and poster."
                    continue

    if best_status == IdentityConsistency.CONSISTENT:
        return IdentityEvidence(
            status=IdentityConsistency.CONSISTENT,
            normalized_poster_names=norm_posters,
            normalized_payee_name=norm_payee,
            match_score=best_score,
            match_rationale=best_rationale,
            is_personal_payee=personal_payee,
        )

    # If no match was found:
    if personal_payee:
        # Crucial requirement: Personal-name payee MUST be treated as AMBIGUOUS, not fraud!
        return IdentityEvidence(
            status=IdentityConsistency.AMBIGUOUS,
            normalized_poster_names=norm_posters,
            normalized_payee_name=norm_payee,
            match_score=0.40,
            match_rationale=(
                f"Payee appears to be an individual/personal name ('{norm_payee}') whereas "
                f"poster displays business branding ({norm_posters}). This is common for sole "
                f"proprietorships and is held as AMBIGUOUS pending further evidence."
            ),
            is_personal_payee=True,
        )

    # Discrepancy between two distinct commercial brand names
    return IdentityEvidence(
        status=IdentityConsistency.CONTRADICTORY,
        normalized_poster_names=norm_posters,
        normalized_payee_name=norm_payee,
        match_score=0.10,
        match_rationale=(
            f"Payee name '{norm_payee}' conflicts with merchant branding {norm_posters}."
        ),
        is_personal_payee=False,
    )
