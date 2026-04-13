"""RecruitMatch API - FastAPI backend for candidate-role matching."""
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.matcher import evaluate_candidate

app = FastAPI(title="RecruitMatch", version="0.1.0")

# Serve static files (recruiter UI)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class MatchRequest(BaseModel):
    client_firm: str
    role_description: str
    candidate_resume: str
    screen_notes: str = ""


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.post("/api/match")
async def match(req: MatchRequest):
    result = await evaluate_candidate(
        client_firm=req.client_firm,
        role_description=req.role_description,
        candidate_resume=req.candidate_resume,
        screen_notes=req.screen_notes,
    )
    return JSONResponse(content=result)


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))}
