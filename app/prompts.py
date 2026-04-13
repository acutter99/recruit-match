"""Prompt templates for LLM-powered analysis and email generation."""


def build_analysis_prompt(
    client_firm: str,
    role_description: str,
    candidate_resume: str,
    screen_notes: str,
) -> str:
    return f"""You are an expert recruiting analyst specializing in legal and financial services staffing.

Analyze the following candidate against the role requirements and return a JSON object with these fields:
- score (0-100): overall fit score
- strengths (list of strings): top 3-5 strengths for this role
- concerns (list of strings): top 2-4 concerns or gaps
- missing_qualifications (list of strings): required qualifications the candidate may lack
- culture_fit (string): brief assessment of likely culture fit
- summary (string): 2-3 sentence overall assessment

CLIENT FIRM: {client_firm}

ROLE DESCRIPTION:
{role_description}

CANDIDATE RESUME:
{candidate_resume}

RECRUITER SCREEN NOTES:
{screen_notes if screen_notes else 'No screen notes provided.'}

Respond with valid JSON only."""


def build_email_prompt(
    client_firm: str,
    role_description: str,
    candidate_resume: str,
    screen_notes: str,
    llm_analysis: dict | None,
) -> str:
    analysis_context = ""
    if llm_analysis:
        strengths = llm_analysis.get("strengths", [])
        summary = llm_analysis.get("summary", "")
        analysis_context = f"\nANALYSIS SUMMARY: {summary}\nKEY STRENGTHS: {', '.join(strengths)}"

    return f"""You are a professional legal/financial services recruiter writing to a client.

Write exactly ONE paragraph (4-6 sentences) to present this candidate to the client. The tone should be professional, confident, and specific. Reference concrete experience from the resume. Do not use generic filler. This should read like a polished candidate presentation email.

CLIENT FIRM: {client_firm}

ROLE BEING FILLED:
{role_description}

CANDIDATE RESUME:
{candidate_resume}

SCREEN NOTES:
{screen_notes if screen_notes else 'No screen notes provided.'}
{analysis_context}

Write the one-paragraph email body only (no subject line, no greeting, no signature)."""
