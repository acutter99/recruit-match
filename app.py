"""
Recruit Match – AI-powered candidate-to-role matching tool.

Input:
- Client firm name
- Role description
- Candidate resume
- Candidate screening notes

Output:
- Match score (0-100)
- Decision label
- Internal reasoning
- One-paragraph client email write-up

To run locally:
    pip install -r requirements.txt
    flask --app app run --reload
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any

from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)


@dataclass
class MatchResult:
    score: int
    decision: str
    reasoning: str
    client_email: str


SYSTEM_PROMPT = """You are an expert legal and financial services recruiter.
You receive:
1) Client firm name
2) Role description
3) Candidate resume
4) Recruiter screening notes

You must:
- Evaluate how strong a match this candidate is for THIS SPECIFIC ROLE,
  focusing on practice area, years of experience, jurisdiction, and relevant deal / case work.
- Think step by step, but only show the final result in the requested JSON format.

Return STRICTLY valid JSON with these keys:
- score: integer from 0 to 100 (higher = better match)
- decision: one of "Strong fit", "Moderate fit", "Weak fit", or "Not a fit"
- reasoning: 3-6 sentence explanation for an internal recruiter, concise and factual
- client_email: a polished 4-6 sentence email paragraph summarizing why you are
  recommending (or not recommending) this candidate. Written in a professional tone
  as if from a specialized legal/financial services recruiter to the hiring partner.
"""

USER_TEMPLATE = """Client firm: {firm}

Role description:
{role}

Candidate resume:
{resume}

Screening notes:
{notes}
"""


def evaluate_match(firm: str, role: str, resume: str, notes: str) -> MatchResult:
    """Call OpenAI to evaluate candidate-role fit."""
    user_prompt = USER_TEMPLATE.format(
        firm=firm.strip(),
        role=role.strip(),
        resume=resume.strip(),
        notes=notes.strip(),
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    content = response.choices[0].message.content
    data: Dict[str, Any] = json.loads(content)

    return MatchResult(
        score=int(data.get("score", 0)),
        decision=str(data.get("decision", "")),
        reasoning=str(data.get("reasoning", "")),
        client_email=str(data.get("client_email", "")),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "firm": request.form.get("firm", ""),
            "role": request.form.get("role", ""),
            "resume": request.form.get("resume", ""),
            "notes": request.form.get("notes", ""),
        }

        if not OPENAI_API_KEY:
            error = "OPENAI_API_KEY is not set. Add it to your .env file."
        elif not form_data["role"] or not form_data["resume"]:
            error = "Role description and candidate resume are required."
        else:
            try:
                result = evaluate_match(**form_data)
            except Exception as e:
                error = f"Error calling model: {e}"

    return render_template("index.html", result=result, error=error, form=form_data)


if __name__ == "__main__":
    app.run(debug=True)
