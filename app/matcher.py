"""Orchestrates candidate evaluation: scoring + LLM analysis + email + write-up generation."""
from app.scoring import compute_baseline_score
from app.llm import call_llm
from app.prompts import build_analysis_prompt, build_email_prompt, build_writeup_prompt


async def evaluate_candidate(
        client_firm: str,
        role_description: str,
        candidate_resume: str,
        screen_notes: str,
        tone: str = "professional",
        custom_instructions: str = "",
) -> dict:
    """Run full evaluation pipeline and return structured result."""

    # Step 1: Baseline keyword/heuristic score
    baseline = compute_baseline_score(role_description, candidate_resume, screen_notes)

    # Step 2: LLM-powered deep analysis (if API key configured)
    analysis_prompt = build_analysis_prompt(
        client_firm, role_description, candidate_resume, screen_notes
    )
    llm_analysis = await call_llm(analysis_prompt)

    # Step 3: Generate client email paragraph
    email_prompt = build_email_prompt(
        client_firm, role_description, candidate_resume, screen_notes, llm_analysis
    )
    email_paragraph = await call_llm(email_prompt)

    # Step 4: Generate candidate write-up with tone control
    writeup_prompt = build_writeup_prompt(
        client_firm, role_description, candidate_resume, screen_notes,
        llm_analysis, tone, custom_instructions
    )
    candidate_writeup = await call_llm(writeup_prompt)

    # Step 5: Determine recommendation
    score = baseline["score"]
    if llm_analysis and "score" in llm_analysis:
        try:
            llm_score = int(llm_analysis.get("score", score))
            score = round((score + llm_score) / 2)
        except (ValueError, TypeError):
            pass

    if score >= 75:
        recommendation = "Strong Match - Recommend presenting to client"
    elif score >= 50:
        recommendation = "Moderate Match - Worth discussing with client"
    else:
        recommendation = "Weak Match - Likely not a fit for this role"

    return {
        "score": score,
        "recommendation": recommendation,
        "baseline": baseline,
        "llm_analysis": llm_analysis,
        "client_email": email_paragraph if email_paragraph else _fallback_email(
            client_firm, candidate_resume, score
        ),
        "candidate_writeup": candidate_writeup if candidate_writeup else _fallback_writeup(
            client_firm, candidate_resume, role_description, score
        ),
        "tone_used": tone,
    }


def _fallback_email(client_firm: str, resume: str, score: int) -> str:
    """Generate a simple fallback email when LLM is unavailable."""
    name_line = resume.strip().split("\n")[0][:60]
    return (
        f"Dear {client_firm} Team, I am pleased to present {name_line} "
        f"for your consideration. Based on our screening process, this "
        f"candidate received a preliminary fit score of {score}/100. "
        f"Please find the attached resume for your review. I look forward "
        f"to discussing this candidate with you at your earliest convenience."
    )


def _fallback_writeup(client_firm: str, resume: str, role_description: str, score: int) -> str:
    """Generate a simple fallback write-up when LLM is unavailable."""
    name_line = resume.strip().split("\n")[0][:60]
    return (
        f"Candidate: {name_line}\n\n"
        f"I am presenting this candidate for the open position at {client_firm}. "
        f"Based on our initial screening, they received a fit score of {score}/100. "
        f"Please review the attached resume and let me know if you would like to "
        f"schedule an interview.\n\n"
        f"Note: For a detailed AI-generated write-up with tone customization, "
        f"please configure an OpenAI or Anthropic API key in your .env file."
    )
