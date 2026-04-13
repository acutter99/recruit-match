"""Sourcing Suggestions Engine - identifies target firms and organizations
to recruit candidates from based on role requirements and geography."""
import re
from app.llm import call_llm
from app.prompts import build_sourcing_prompt

# Mapping of practice areas / role types to typical employer categories
SOURCING_MAP = {
    "litigation": {
        "firm_types": ["litigation boutiques", "insurance defense firms", "plaintiff firms",
                       "general practice firms with litigation departments", "government AG offices"],
        "org_types": ["county/city attorney offices", "public defender offices",
                      "corporate legal departments", "insurance carriers"],
    },
    "insurance defense": {
        "firm_types": ["insurance defense firms", "litigation boutiques",
                       "firms on insurance panel counsel lists", "general practice firms"],
        "org_types": ["insurance carriers (claims/litigation units)", "TPAs",
                      "corporate risk management departments"],
    },
    "real estate": {
        "firm_types": ["real estate boutiques", "transactional firms",
                       "full-service firms with real estate groups", "title companies"],
        "org_types": ["REITs", "commercial developers", "property management companies",
                      "title insurance companies", "mortgage lenders", "banks (real estate lending)"],
    },
    "corporate": {
        "firm_types": ["corporate/transactional firms", "M&A boutiques",
                       "full-service firms with corporate groups", "venture/startup firms"],
        "org_types": ["in-house legal departments", "private equity firms",
                      "venture capital firms", "investment banks", "Big 4 advisory"],
    },
    "family law": {
        "firm_types": ["family law boutiques", "domestic relations firms",
                       "general practice firms with family law groups", "legal aid organizations"],
        "org_types": ["family court", "mediation centers", "child advocacy organizations",
                      "domestic violence legal services"],
    },
    "employment law": {
        "firm_types": ["employment/labor boutiques", "management-side firms",
                       "plaintiff employment firms", "union-side firms"],
        "org_types": ["EEOC", "state labor departments", "corporate HR/legal departments",
                      "unions", "staffing companies (legal)"],
    },
    "workers compensation": {
        "firm_types": ["workers comp defense firms", "insurance defense firms",
                       "claimant-side workers comp firms"],
        "org_types": ["insurance carriers (WC divisions)", "state workers comp boards",
                      "TPAs", "self-insured employers"],
    },
    "bankruptcy": {
        "firm_types": ["bankruptcy boutiques", "restructuring firms",
                       "creditor-side firms", "debtor-side firms"],
        "org_types": ["bankruptcy court", "US Trustee offices", "banks (workout groups)",
                      "turnaround consulting firms", "distressed debt funds"],
    },
    "compliance": {
        "firm_types": ["regulatory/compliance firms", "white collar defense firms",
                       "financial services firms"],
        "org_types": ["bank compliance departments", "broker-dealer compliance",
                      "fintech companies", "SEC/FINRA", "state regulators", "Big 4 advisory"],
    },
    "financial services": {
        "firm_types": ["financial services law firms", "securities boutiques",
                       "regulatory firms", "hedge fund counsel"],
        "org_types": ["investment banks", "commercial banks", "asset managers",
                      "hedge funds", "private equity firms", "insurance companies",
                      "fintech companies", "broker-dealers"],
    },
    "accounting": {
        "firm_types": ["Big 4 firms", "regional CPA firms", "national mid-tier firms",
                       "forensic accounting firms", "tax boutiques"],
        "org_types": ["corporate accounting departments", "banks (finance teams)",
                      "insurance companies", "government (audit/finance)",
                      "consulting firms", "PE portfolio companies"],
    },
}


async def generate_sourcing_suggestions(
    role_description: str,
    location: str,
    client_firm: str = "",
) -> dict:
    """Generate a list of target firms/orgs to source candidates from."""

    # Step 1: Detect practice areas from role description
    detected_areas = _detect_practice_areas(role_description)

    # Step 2: Build baseline suggestions from our mapping
    baseline_targets = _build_baseline_targets(detected_areas, location, client_firm)

    # Step 3: Use LLM for deeper, location-specific suggestions
    sourcing_prompt = build_sourcing_prompt(role_description, location, client_firm)
    llm_suggestions = await call_llm(sourcing_prompt)

    return {
        "detected_practice_areas": detected_areas,
        "location": location,
        "baseline_targets": baseline_targets,
        "llm_suggestions": llm_suggestions,
    }


def _detect_practice_areas(role_description: str) -> list:
    """Identify which practice areas are relevant to the role."""
    role_lower = role_description.lower()
    detected = []
    for area in SOURCING_MAP:
        if area in role_lower:
            detected.append(area)

    # Check for additional signals
    signal_map = {
        "litigation": ["litigat", "trial", "discovery", "deposition", "courtroom"],
        "insurance defense": ["insurance", "carrier", "coverage", "claims"],
        "real estate": ["real estate", "property", "title", "mortgage", "closing", "zoning"],
        "corporate": ["m&a", "merger", "acquisition", "securities", "venture", "private equity", "transactional"],
        "family law": ["family", "divorce", "custody", "domestic", "matrimonial"],
        "employment law": ["employment", "labor", "eeoc", "discrimination", "wrongful termination"],
        "workers compensation": ["workers comp", "work comp", "occupational injury"],
        "bankruptcy": ["bankrupt", "restructur", "creditor", "debtor", "chapter 11"],
        "compliance": ["compliance", "regulatory", "aml", "bsa", "kyc"],
        "financial services": ["financial", "banking", "investment", "asset management", "fintech"],
        "accounting": ["accounting", "audit", "tax", "cpa", "financial reporting", "gaap"],
    }
    for area, signals in signal_map.items():
        if area not in detected:
            for signal in signals:
                if signal in role_lower:
                    detected.append(area)
                    break

    if not detected:
        detected = ["corporate"]  # default fallback

    return list(set(detected))


def _build_baseline_targets(practice_areas: list, location: str, exclude_firm: str) -> dict:
    """Build structured sourcing targets from our mapping."""
    firm_types = set()
    org_types = set()

    for area in practice_areas:
        if area in SOURCING_MAP:
            firm_types.update(SOURCING_MAP[area]["firm_types"])
            org_types.update(SOURCING_MAP[area]["org_types"])

    # Add location context
    location_note = f"Focus sourcing in and around {location}" if location else "No geographic preference specified"

    return {
        "target_firm_types": sorted(firm_types),
        "target_organization_types": sorted(org_types),
        "location_guidance": location_note,
        "exclude_client": exclude_firm if exclude_firm else None,
        "sourcing_tips": _get_sourcing_tips(practice_areas, location),
    }


def _get_sourcing_tips(practice_areas: list, location: str) -> list:
    """Return actionable sourcing tips based on practice area."""
    tips = [
        f"Search LinkedIn for attorneys/professionals in {location} with relevant practice area keywords",
        "Check local/state bar association directories for attorneys in the practice area",
        "Review recent court filings and case dockets for active practitioners in the area",
        "Look at local business journal awards (e.g. Best Lawyers, Super Lawyers) for the practice area",
    ]
    if any(a in practice_areas for a in ["litigation", "insurance defense", "workers compensation"]):
        tips.append("Search PACER/state court dockets for attorneys actively filing in this area")
        tips.append("Check insurance carrier panel counsel lists for the region")
    if any(a in practice_areas for a in ["corporate", "real estate", "financial services"]):
        tips.append("Review recent deal announcements and transaction tombstones in the market")
        tips.append("Check local commercial real estate and M&A deal lists")
    if any(a in practice_areas for a in ["compliance", "financial services", "accounting"]):
        tips.append("Look at regulatory filings and enforcement actions for active professionals")
        tips.append("Check Big 4 and mid-tier firm alumni networks in the region")
    if location:
        tips.append(f"Search for professionals who recently relocated TO {location} - they may be settling in")
        tips.append(f"Look at firms in {location} that recently lost partners or had layoffs")
    return tips
