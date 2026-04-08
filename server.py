"""
Email Triage Hub – FastAPI Server
Implements the OpenEnv REST interface:
  GET  /health          – liveness probe
  GET  /ping            – alias for health (openenv validator)
  POST /reset           – start a new episode
  POST /step            – submit one triage action
  GET  /state           – inspect full internal state
  GET  /tasks           – list available tasks
  GET  /openenv.yaml    – serve environment manifest
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from env.email_triage_env import EmailTriageEnv
from env.models import (
    EnvironmentState,
    ResetRequest,
    StepResult,
    TriageAction,
)
from env.tasks import TASK_REGISTRY

# App setup 

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

# Global environment instance (single session per container)

_env: Optional[EmailTriageEnv] = None


#  Health / discovery

@app.get("/health", tags=["system"])
@app.get("/ping", tags=["system"])
async def health():
    """Liveness probe. Returns 200 OK with environment info."""
    return {
        "status": "ok",
        "env": "email-triage-hub",
        "version": "1.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
    }


@app.get("/tasks", tags=["system"])
async def list_tasks():
    """List all available tasks with metadata."""
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
    """Serve the openenv.yaml manifest file."""
    manifest_path = Path("openenv.yaml")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return Response(
        content=manifest_path.read_text(),
        media_type="application/x-yaml",
    )


# Core OpenEnv endpoints 

@app.post("/reset", response_model=StepResult, tags=["openenv"])
async def reset(request: ResetRequest = None):  # type: ignore[assignment]
    """
    Start a new episode.

    Body (JSON):  { "task": "priority_sort" }   (default: priority_sort)
    Returns the first observation plus reward=0.0, done=False.
    """
    global _env

    if request is None:
        request = ResetRequest()

    if request.task not in TASK_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown task '{request.task}'. "
                f"Valid tasks: {list(TASK_REGISTRY.keys())}"
            ),
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
    """
    Submit a triage action for the current email.

    The action must target the current email (`email_id` should match).
    Returns the next observation, per-step reward, and done flag.
    """
    global _env

    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )

    obs, reward, done, info = _env.step(action)

    return StepResult(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
async def state():
    """Return the full internal state of the environment (for debugging)."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env.state()


# Entry point 

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
