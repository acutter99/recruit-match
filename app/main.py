"""RecruitMatch API - FastAPI backend for candidate-role matching."""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from app.matcher import evaluate_candidate
from app.sourcing import generate_sourcing_suggestions

app = FastAPI(title="RecruitMatch", version="0.3.0")

# Serve static files (recruiter UI)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- Data persistence (JSON file) ---
DATA_FILE = Path("data/clients.json")


def _load_data() -> dict:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return {"clients": {}}


def _save_data(data: dict):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2))


# --- Pydantic models ---
class MatchRequest(BaseModel):
    client_firm: str
    role_description: str
    candidate_resume: str
    screen_notes: str = ""


class SourceRequest(BaseModel):
    role_description: str
    location: str
    client_firm: str = ""


class ClientCreate(BaseModel):
    name: str
    notes: str = ""


class RoleCreate(BaseModel):
    title: str
    description: str


# --- Page routes ---
@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


# --- Client CRUD ---
@app.get("/api/clients")
async def list_clients():
    data = _load_data()
    clients = []
    for cid, c in data["clients"].items():
        clients.append({"id": cid, "name": c["name"], "notes": c.get("notes", ""), "created_at": c["created_at"], "role_count": len(c.get("roles", {}))})
    clients.sort(key=lambda x: x["name"].lower())
    return JSONResponse(content=clients)


@app.post("/api/clients")
async def create_client(req: ClientCreate):
    data = _load_data()
    cid = str(uuid.uuid4())[:8]
    data["clients"][cid] = {
        "name": req.name,
        "notes": req.notes,
        "created_at": datetime.utcnow().isoformat(),
        "roles": {},
    }
    _save_data(data)
    return JSONResponse(content={"id": cid, "name": req.name})


@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    data = _load_data()
    c = data["clients"].get(client_id)
    if not c:
        return JSONResponse(content={"error": "Client not found"}, status_code=404)
    roles = []
    for rid, r in c.get("roles", {}).items():
        roles.append({"id": rid, "title": r["title"], "description": r["description"], "created_at": r["created_at"]})
    roles.sort(key=lambda x: x["created_at"], reverse=True)
    return JSONResponse(content={"id": client_id, "name": c["name"], "notes": c.get("notes", ""), "roles": roles})


@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str):
    data = _load_data()
    if client_id in data["clients"]:
        del data["clients"][client_id]
        _save_data(data)
    return JSONResponse(content={"ok": True})


# --- Role CRUD ---
@app.post("/api/clients/{client_id}/roles")
async def create_role(client_id: str, req: RoleCreate):
    data = _load_data()
    c = data["clients"].get(client_id)
    if not c:
        return JSONResponse(content={"error": "Client not found"}, status_code=404)
    rid = str(uuid.uuid4())[:8]
    c.setdefault("roles", {})[rid] = {
        "title": req.title,
        "description": req.description,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_data(data)
    return JSONResponse(content={"id": rid, "title": req.title})


@app.delete("/api/clients/{client_id}/roles/{role_id}")
async def delete_role(client_id: str, role_id: str):
    data = _load_data()
    c = data["clients"].get(client_id)
    if c and role_id in c.get("roles", {}):
        del c["roles"][role_id]
        _save_data(data)
    return JSONResponse(content={"ok": True})


# --- Matching & Sourcing ---
@app.post("/api/match")
async def match(req: MatchRequest):
    result = await evaluate_candidate(
        client_firm=req.client_firm,
        role_description=req.role_description,
        candidate_resume=req.candidate_resume,
        screen_notes=req.screen_notes,
    )
    return JSONResponse(content=result)


@app.post("/api/source")
async def source(req: SourceRequest):
    result = await generate_sourcing_suggestions(
        role_description=req.role_description,
        location=req.location,
        client_firm=req.client_firm,
    )
    return JSONResponse(content=result)


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))}
