"""Baseline keyword and heuristic scoring (no LLM required)."""
import re
from collections import Counter


def compute_baseline_score(
    role_description: str,
    candidate_resume: str,
    screen_notes: str,
) -> dict:
    """Return a dict with score (0-100), matched keywords, and gaps."""
    role_lower = role_description.lower()
    resume_lower = candidate_resume.lower()
    notes_lower = screen_notes.lower()
    combined = resume_lower + " " + notes_lower

    # Extract meaningful keywords from role description
    role_keywords = _extract_keywords(role_lower)

    matched = []
    missing = []
    for kw in role_keywords:
        if kw in combined:
            matched.append(kw)
        else:
            missing.append(kw)

    # Score components
    keyword_ratio = len(matched) / max(len(role_keywords), 1)

    # Check for years of experience mentions
    exp_score = _experience_score(role_lower, combined)

    # Check for education/bar admission signals
    edu_score = _education_score(role_lower, combined)

    # Screen notes sentiment bonus
    notes_bonus = _notes_bonus(screen_notes)

    raw_score = (
        keyword_ratio * 50  # keyword match worth up to 50 pts
        + exp_score * 20     # experience worth up to 20 pts
        + edu_score * 15     # education worth up to 15 pts
        + notes_bonus * 15   # screen notes worth up to 15 pts
    )
    score = min(100, max(0, round(raw_score)))

    return {
        "score": score,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_match_pct": round(keyword_ratio * 100, 1),
        "experience_signal": exp_score,
        "education_signal": edu_score,
        "notes_signal": notes_bonus,
    }


def _extract_keywords(text: str) -> list:
    """Pull out meaningful multi-word and single-word terms."""
    # Common legal/professional terms to look for
    domain_terms = [
        "litigation", "insurance defense", "workers compensation",
        "real estate", "transactional", "corporate", "compliance",
        "family law", "estate planning", "intellectual property",
        "employment law", "labor law", "bankruptcy", "tax",
        "healthcare", "regulatory", "commercial", "contract",
        "discovery", "deposition", "trial", "arbitration",
        "mediation", "due diligence", "mergers", "acquisitions",
        "securities", "private equity", "venture capital",
        "j.d.", "bar admission", "licensed", "partner",
        "associate", "of counsel", "senior", "junior",
        "python", "sql", "excel", "financial modeling",
        "accounting", "cpa", "cfa", "series 7", "finra",
        "underwriting", "risk management", "portfolio",
    ]
    found = [t for t in domain_terms if t in text]

    # Also grab capitalized phrases and numbers+years patterns
    words = re.findall(r'\b[a-z]{4,}\b', text)
    freq = Counter(words)
    top = [w for w, c in freq.most_common(20)
           if w not in _STOPWORDS and len(w) > 3]

    return list(set(found + top[:15]))


def _experience_score(role: str, candidate: str) -> float:
    """Check if candidate meets experience requirements."""
    req_years = re.findall(r'(\d+)\+?\s*years?', role)
    cand_years = re.findall(r'(\d+)\+?\s*years?', candidate)
    if not req_years:
        return 0.7  # no requirement stated, neutral
    if not cand_years:
        return 0.3  # requirement stated but no experience found
    req_max = max(int(y) for y in req_years)
    cand_max = max(int(y) for y in cand_years)
    if cand_max >= req_max:
        return 1.0
    elif cand_max >= req_max * 0.7:
        return 0.6
    return 0.2


def _education_score(role: str, candidate: str) -> float:
    """Check education and credential alignment."""
    edu_terms = ["j.d.", "juris doctor", "bar", "admitted",
                 "mba", "cpa", "cfa", "bachelor", "master", "ph.d."]
    role_edu = [t for t in edu_terms if t in role]
    if not role_edu:
        return 0.7
    cand_edu = [t for t in edu_terms if t in candidate]
    if not cand_edu:
        return 0.2
    overlap = set(role_edu) & set(cand_edu)
    return len(overlap) / len(role_edu) if role_edu else 0.5


def _notes_bonus(notes: str) -> float:
    """Score based on positive/negative signals in screen notes."""
    if not notes.strip():
        return 0.5
    pos = ["strong", "excellent", "impressive", "great", "solid",
           "articulate", "motivated", "enthusiastic", "polished",
           "experienced", "knowledgeable", "professional", "good fit"]
    neg = ["weak", "concern", "red flag", "not a fit", "overqualified",
           "underqualified", "lacks", "poor", "difficult", "hesitant",
           "uncommitted", "flight risk"]
    notes_l = notes.lower()
    pos_count = sum(1 for p in pos if p in notes_l)
    neg_count = sum(1 for n in neg if n in notes_l)
    net = pos_count - neg_count
    return max(0.0, min(1.0, 0.5 + net * 0.15))


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have",
    "will", "been", "were", "are", "was", "has", "had", "not",
    "but", "what", "all", "can", "her", "his", "our", "out",
    "about", "which", "their", "them", "then", "than", "into",
    "could", "would", "should", "also", "some", "other", "more",
    "very", "just", "must", "including", "such", "experience",
    "work", "ability", "strong", "position", "role", "team",
}
