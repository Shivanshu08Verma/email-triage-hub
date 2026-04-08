"""
server/app.py – Email Triage Hub FastAPI Application

Full OpenEnv-compliant server with all required endpoints:
  GET  /health    → {"status": "healthy"}
  GET  /ping      → alias for health
  GET  /metadata  → env name + description
  GET  /schema    → action / observation / state schemas
  POST /mcp       → JSON-RPC stub (required by openenv validate)
  POST /reset     → start a new episode
  POST /step      → submit one triage action
  GET  /state     → full internal state
  GET  /tasks     → list available tasks
  GET  /openenv.yaml → serve manifest

Entry point: main() — called by `uv run server` via pyproject.toml [project.scripts]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from project root when running as `server.app`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from env.email_triage_env import EmailTriageEnv
from env.models import (
    EnvironmentState,
    ResetRequest,
    StepResult,
    TriageAction,
)
from env.tasks import TASK_REGISTRY

# App

app = FastAPI(
    title="Email Triage Hub",
    description=(
        "OpenEnv environment where AI agents learn to triage realistic business emails: "
        "priority classification, department routing, spam detection, and response drafting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env state

_env: Optional[EmailTriageEnv] = None

# System / discovery endpoints 

@app.get("/health", tags=["system"])
@app.get("/ping", tags=["system"])
async def health():
    """
    Liveness probe.
    Returns {"status": "healthy"} — required by openenv validate.
    """
    return {
        "status": "healthy",
        "env": "email-triage-hub",
        "version": "1.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
    }


@app.get("/metadata", tags=["system"])
async def metadata():
    """
    Environment metadata — required by openenv validate runtime check.
    Must return at minimum: name (str) and description (str).
    """
    return {
        "name": "email-triage-hub",
        "description": (
            "A realistic business email triage environment where AI agents learn to "
            "classify priority (urgent/normal/low), route emails to the correct department, "
            "detect spam/phishing, and draft professional responses. "
            "Three tasks of increasing difficulty: easy → medium → hard."
        ),
        "version": "1.0.0",
        "tasks": [
            {
                "id": spec.id,
                "name": spec.name,
                "difficulty": spec.difficulty,
            }
            for spec in TASK_REGISTRY.values()
        ],
    }


@app.get("/schema", tags=["system"])
async def schema():
    """
    Returns action, observation, and state JSON schemas.
    Required by openenv validate runtime check.
    """
    return {
        "action": {
            "type": "object",
            "description": "Agent triage decision for one email",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "ID of the email being triaged (e.g. 'E001')",
                    "required": True,
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "normal", "low"],
                    "description": "Required for non-spam emails in all tasks",
                },
                "department": {
                    "type": "string",
                    "enum": ["IT", "HR", "Sales", "Support", "Finance", "Legal", "Management"],
                    "description": "Required for tasks 2 and 3 (non-spam emails)",
                },
                "is_spam": {
                    "type": "boolean",
                    "description": "True if phishing/spam. Required for task 3",
                },
                "response_draft": {
                    "type": "string",
                    "description": "50-200 word professional reply. Required for urgent non-spam in task 3",
                },
            },
        },
        "observation": {
            "type": "object",
            "description": "Current email + inbox progress visible to the agent",
            "properties": {
                "current_email": {
                    "type": "object",
                    "nullable": True,
                    "description": "Email to triage next; null when episode ends",
                    "properties": {
                        "email_id":       {"type": "string"},
                        "sender":         {"type": "string"},
                        "subject":        {"type": "string"},
                        "body":           {"type": "string"},
                        "timestamp":      {"type": "string", "format": "date-time"},
                        "has_attachment": {"type": "boolean"},
                    },
                },
                "inbox_summary": {
                    "type": "object",
                    "properties": {
                        "total_emails":     {"type": "integer"},
                        "processed":        {"type": "integer"},
                        "remaining":        {"type": "integer"},
                        "task_name":        {"type": "string"},
                        "task_description": {"type": "string"},
                    },
                },
                "step_feedback":     {"type": "string", "nullable": True},
                "task_description":  {"type": "string"},
                "requirements":      {"type": "array", "items": {"type": "string"}},
                "available_actions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "description": "Full internal environment state (for debugging/inspection)",
            "properties": {
                "task":                  {"type": "string"},
                "current_email_index":   {"type": "integer"},
                "total_emails":          {"type": "integer"},
                "processed_actions":     {"type": "array"},
                "cumulative_reward":     {"type": "number"},
                "done":                  {"type": "boolean"},
            },
        },
    }


@app.post("/mcp", tags=["system"])
async def mcp_endpoint(request: Request):
    """
    MCP (Model Context Protocol) JSON-RPC stub.
    Required by openenv validate runtime check.
    Returns a valid JSON-RPC 2.0 response.
    """
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        body = {}

    method = body.get("method", "")
    req_id = body.get("id", 1)

    # Handle standard JSON-RPC MCP methods
    if method == "initialize":
        result: Dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "email-triage-hub", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Start a new triage episode",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"task": {"type": "string"}},
                    },
                },
                {
                    "name": "step",
                    "description": "Submit a triage action for the current email",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"email_id": {"type": "string"}},
                        "required": ["email_id"],
                    },
                },
            ]
        }
    else:
        # Unknown method — return empty result (valid JSON-RPC)
        result = {}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


@app.get("/tasks", tags=["system"])
async def list_tasks():
    """List all available tasks with full metadata."""
    return {
        "tasks": [
            {
                "id": spec.id,
                "name": spec.name,
                "difficulty": spec.difficulty,
                "max_steps": spec.max_steps,
                "max_total_reward": spec.max_total_reward,
                "success_threshold": spec.success_threshold,
                "description": spec.description,
                "requirements": spec.requirements,
            }
            for spec in TASK_REGISTRY.values()
        ]
    }


@app.get("/openenv.yaml", tags=["system"])
async def serve_manifest():
    """Serve the openenv.yaml manifest."""
    manifest_path = _project_root / "openenv.yaml"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return Response(content=manifest_path.read_text(), media_type="application/x-yaml")


# Core OpenEnv endpoints 

@app.post("/reset", response_model=StepResult, tags=["openenv"])
async def reset(request: Optional[ResetRequest] = None):
    """
    Start a new episode.
    Body: {"task": "priority_sort"}  (default: priority_sort)
    """
    global _env

    if request is None:
        request = ResetRequest()

    if request.task not in TASK_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{request.task}'. Valid: {list(TASK_REGISTRY.keys())}",
        )

    _env = EmailTriageEnv(task=request.task)
    obs = _env.reset()

    return StepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={
            "task": request.task,
            "total_emails": len(_env.email_ids),
            "max_steps": TASK_REGISTRY[request.task].max_steps,
            "max_total_reward": TASK_REGISTRY[request.task].max_total_reward,
        },
    )


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(action: TriageAction):
    """Submit a triage action for the current email."""
    global _env

    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )

    obs, reward, done, info = _env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
async def state():
    """Return full internal environment state."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env.state()


# Entry point

def main() -> None:
    """
    Entry point called by:
      - `uv run server`           (via pyproject.toml [project.scripts])
      - `python -m server.app`
      - `python server/app.py`
    """
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
