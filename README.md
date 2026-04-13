# RecruitMatch

AI-powered candidate-to-role matching tool built for recruiters in legal and financial services.

## What It Does

1. **Paste the role** - Enter the client firm name and full role description
2. **Paste the candidate** - Enter resume text and your screen notes
3. **Get instant results** - Fit score (0-100), matched/missing keywords, strengths, concerns
4. **Copy the email** - One-paragraph client presentation write-up, ready to send

## Quick Start

```bash
# Clone the repo
git clone https://github.com/acutter99/recruit-match.git
cd recruit-match

# Set up environment
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI or Anthropic API key (optional)

# Run the app
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser.

## How Scoring Works

- **Keyword matching (50 pts)** - Domain terms from the role description matched against resume
- **Experience check (20 pts)** - Years of experience comparison
- **Education/credentials (15 pts)** - Bar admission, degrees, certifications
- **Screen notes sentiment (15 pts)** - Positive/negative signals from your notes

If an LLM API key is configured, the app also runs a deep analysis and averages the LLM score with the baseline.

## LLM Support

Set ONE of these in your `.env` file:
- `OPENAI_API_KEY` - Uses GPT-4o-mini by default
- `ANTHROPIC_API_KEY` - Uses Claude Sonnet by default

The app works without any API key using baseline keyword scoring only.

## Project Structure

```
app/
  main.py          # FastAPI endpoints
  matcher.py       # Evaluation orchestrator
  scoring.py       # Baseline keyword/heuristic scoring
  prompts.py       # LLM prompt templates
  llm.py           # OpenAI + Anthropic API integration
  static/
    index.html     # Recruiter web UI
.env.example       # Environment template
requirements.txt   # Python dependencies
```

## Roadmap

- [ ] PDF/DOCX resume upload and parsing
- [ ] Score weighting by practice area (litigation, transactional, etc.)
- [ ] Candidate and job history database
- [ ] Client email formatting options
- [ ] Batch candidate screening
