"""Prompt templates for LLM-powered analysis, email generation, and candidate write-ups."""


# Tone definitions for candidate write-ups
TONE_INSTRUCTIONS = {
    "professional": "Write in a polished, formal business tone. Use precise language, avoid slang, and maintain a confident authoritative voice. This should sound like a senior recruiter at a top-tier staffing firm.",
    "conversational": "Write in a warm, approachable tone as if you're talking to the client over coffee. Be personable but still knowledgeable. Use natural language, contractions are fine, and let your enthusiasm for the candidate come through.",
    "executive": "Write in a concise, high-level executive briefing style. Lead with impact and results. Every sentence should deliver value. No filler, no fluff - think C-suite communication. Bullet points or short punchy sentences work well.",
    "consultative": "Write as a trusted advisor giving strategic counsel. Frame the candidate in terms of business impact and ROI. Reference market conditions and why this candidate stands out. Position yourself as a market expert, not just a recruiter.",
    "enthusiastic": "Write with genuine energy and excitement about this candidate. Be upbeat and positive while remaining credible. Convey urgency - this is someone the client needs to move on quickly. Use strong action words and paint a compelling picture.",
}


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


def build_writeup_prompt(
        client_firm: str,
        role_description: str,
        candidate_resume: str,
        screen_notes: str,
        llm_analysis: dict | None,
        tone: str = "professional",
        custom_instructions: str = "",
) -> str:
    """Build a prompt for generating a candidate write-up with tone control."""
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["professional"])

    analysis_context = ""
    if llm_analysis:
        strengths = llm_analysis.get("strengths", [])
        concerns = llm_analysis.get("concerns", [])
        summary = llm_analysis.get("summary", "")
        score = llm_analysis.get("score", "N/A")
        analysis_context = f"""\nANALYSIS RESULTS:
- Fit Score: {score}/100
- Summary: {summary}
- Key Strengths: {', '.join(strengths)}
- Concerns: {', '.join(concerns)}"""

    custom_section = ""
    if custom_instructions:
        custom_section = f"\nADDITIONAL INSTRUCTIONS FROM RECRUITER:\n{custom_instructions}\n"

    return f"""You are an experienced recruiter at a specialized legal and financial services staffing firm. You are writing a candidate presentation write-up to send to your client.

TONE AND STYLE:
{tone_instruction}

Your write-up should:
1. Open with a strong hook about why this candidate is worth the client's attention
2. Highlight 2-3 specific experiences or qualifications from the resume that align with the role
3. Address any potential concerns proactively if relevant
4. Include a clear call to action (suggest an interview, next steps)
5. Be 2-4 paragraphs total
6. Feel like it was written by a real recruiter who knows this candidate, not generated by AI
{custom_section}
CLIENT FIRM: {client_firm}

ROLE BEING FILLED:
{role_description}

CANDIDATE RESUME:
{candidate_resume}

RECRUITER SCREEN NOTES:
{screen_notes if screen_notes else 'No screen notes provided.'}
{analysis_context}

Write the candidate presentation write-up now. Output the write-up text only, no headers or labels."""


def build_sourcing_prompt(
        role_description: str,
        location: str,
        client_firm: str = "",
) -> str:
    exclude_note = f"\nIMPORTANT: Do NOT include {client_firm} in your suggestions - that is the hiring client." if client_firm else ""

    return f"""You are an expert legal and financial services recruiter with deep knowledge of firm landscapes across the United States.

Given the following role description and geographic location, identify specific firms, companies, and organizations where qualified candidates for this role are likely currently working. These are sourcing targets - places to recruit FROM.

For each suggestion, provide:
- name: The firm or organization name
- type: Category (e.g. "Law Firm", "Insurance Carrier", "Bank", "Corporate Legal Dept", etc.)
- relevance: Why candidates from here would be a good fit (1 sentence)
- size_tier: "Large", "Mid-size", or "Small/Boutique"

Return a JSON object with these fields:
- target_firms (list of objects): 8-12 specific named firms/companies in or near the location
- target_categories (list of strings): 5-8 broader categories of employers to search
- linkedin_search_keywords (list of strings): 5-8 keyword combinations for LinkedIn searches
- job_boards_to_monitor (list of strings): 3-5 job boards or listing sites to watch for competitor openings
- networking_suggestions (list of strings): 3-5 local events, bar associations, or groups to engage
- sourcing_strategy (string): 2-3 sentence recommended sourcing approach

ROLE DESCRIPTION:
{role_description}

GEOGRAPHIC LOCATION: {location}
{exclude_note}

Respond with valid JSON only."""
