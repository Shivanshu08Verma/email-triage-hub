"""
inference.py – Baseline inference script for the Email Triage Hub OpenEnv environment.

Runs all three tasks (priority_sort, department_routing, full_triage) sequentially
using an LLM via the OpenAI-compatible API client.

Required environment variables:
  API_BASE_URL     - LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME       - Model identifier  (default: gpt-4o-mini)
  HF_TOKEN         - API key (no default, must be set explicitly)
  LOCAL_IMAGE_NAME - Optional Docker image name if using from_docker_image()
  ENV_URL          - Email Triage Hub server URL (default: http://localhost:8000)

Stdout format:  [START] / [STEP] / [END]  structured logs (required by evaluator).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# Configuration

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")                   
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "email-triage-hub"
SUCCESS_SCORE_THRESHOLD = 0.5

# Per-task configuration
TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "priority_sort": {
        "max_steps": 10,
        "max_total_reward": 1.0,
    },
    "department_routing": {
        "max_steps": 20,
        "max_total_reward": 1.0,
    },
    "full_triage": {
        "max_steps": 30,
        "max_total_reward": 1.0,
    },
}

TASKS = list(TASK_CONFIG.keys())

# Structured logging (required format)

def log_start(*, task: str, env: str, model: str) -> None:
    """Emit [START] log line."""
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    *,
    step: int,
    action: Any,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Emit [STEP] log line."""
    action_str = json.dumps(action) if not isinstance(action, str) else action
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log line."""
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# Environment HTTP client

async def env_reset(client: httpx.AsyncClient, task: str) -> Dict[str, Any]:
    """POST /reset and return the parsed JSON response."""
    resp = await client.post(
        f"{ENV_URL}/reset",
        json={"task": task},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def env_step(
    client: httpx.AsyncClient, action: Dict[str, Any]
) -> Dict[str, Any]:
    """POST /step and return the parsed JSON response."""
    resp = await client.post(
        f"{ENV_URL}/step",
        json=action,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# LLM interaction

def _build_system_prompt(task: str) -> str:
    task_instructions = {
        "priority_sort": (
            "You are an expert email triage assistant. Your sole job is to classify "
            "the priority of each business email as 'urgent', 'normal', or 'low'.\n\n"
            "Definitions:\n"
            "  urgent – requires immediate action today (outages, security incidents, "
            "same-day deadlines with business impact)\n"
            "  normal – important, should be handled within a few hours to a day\n"
            "  low    – informational, routine, optional, non-time-sensitive\n\n"
            "Respond ONLY with valid JSON. No explanation, no markdown fences.\n"
            "Format:\n"
            '{"email_id": "<id>", "priority": "urgent|normal|low"}'
        ),
        "department_routing": (
            "You are an expert email triage assistant. Classify the priority AND route "
            "each email to the correct department.\n\n"
            "Priority levels: urgent | normal | low\n"
            "Departments: IT | HR | Sales | Support | Finance | Legal | Management\n\n"
            "Department guide:\n"
            "  IT         – systems, servers, software, security, infrastructure\n"
            "  HR         – employees, hiring, onboarding, accidents, training\n"
            "  Sales      – leads, partnerships, deals, contracts, renewals\n"
            "  Support    – customer complaints, billing disputes, refunds\n"
            "  Finance    – invoices, payments, budgets, procurement\n"
            "  Legal      – NDAs, contracts, compliance, regulatory\n"
            "  Management – executive decisions, strategy\n\n"
            "Respond ONLY with valid JSON. No explanation, no markdown fences.\n"
            "Format:\n"
            '{"email_id": "<id>", "priority": "urgent|normal|low", '
            '"department": "IT|HR|Sales|Support|Finance|Legal|Management"}'
        ),
        "full_triage": (
            "You are an expert email triage assistant performing complete email triage.\n\n"
            "For EVERY email:\n"
            "  1. Set is_spam: true if phishing / scam / unsolicited (suspicious domain, "
            "prize claims, credential harvesting). false otherwise.\n\n"
            "For NON-SPAM emails:\n"
            "  2. priority: urgent | normal | low\n"
            "  3. department: IT | HR | Sales | Support | Finance | Legal | Management\n\n"
            "For URGENT non-spam emails:\n"
            "  4. response_draft: 50–200 word professional reply that:\n"
            "       a) Acknowledges the issue with empathy\n"
            "       b) States concrete next actions\n"
            "       c) Uses professional business tone\n\n"
            "Spam signals: mismatched domains, ALL CAPS urgency, prize claims, "
            "requests for SSN/bank/card details, suspicious links.\n\n"
            "Respond ONLY with valid JSON. No explanation, no markdown fences.\n"
            "Format:\n"
            '{"email_id":"<id>","is_spam":false,"priority":"urgent|normal|low",'
            '"department":"IT|HR|...","response_draft":"..."}'
        ),
    }
    return task_instructions.get(task, task_instructions["priority_sort"])


def _build_user_prompt(
    observation: Dict[str, Any],
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    """Format observation into a user prompt for the LLM."""
    lines: List[str] = []

    if history:
        lines.append("=== Previous triage decisions ===")
        lines.extend(history[-5:])  # last 5 for context
        lines.append("")

    summary = observation.get("inbox_summary", {})
    lines.append(
        f"Progress: {summary.get('processed', 0)}/{summary.get('total_emails', '?')} "
        f"emails processed. Remaining: {summary.get('remaining', '?')}."
    )

    if last_reward > 0 and step > 1:
        lines.append(f"Last step reward: {last_reward:.4f}")

    feedback = observation.get("step_feedback")
    if feedback:
        lines.append(f"Feedback: {feedback}")

    lines.append("")

    email = observation.get("current_email")
    if email is None:
        lines.append("No more emails. Episode complete.")
        return "\n".join(lines)

    lines.append("=== CURRENT EMAIL TO TRIAGE ===")
    lines.append(f"Email ID : {email.get('email_id', '?')}")
    lines.append(f"From     : {email.get('sender', '?')}")
    lines.append(f"Subject  : {email.get('subject', '?')}")
    lines.append(f"Time     : {email.get('timestamp', '?')}")
    if email.get("has_attachment"):
        lines.append("Attachment: YES")
    lines.append("")
    lines.append("Body:")
    lines.append(email.get("body", "(empty)"))
    lines.append("")
    lines.append(f"Email ID to use in your JSON: {email.get('email_id', '?')}")

    return "\n".join(lines)


def _call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    task: str,
) -> str:
    """Call the LLM and return raw text response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def _parse_action(llm_response: str, email_id: str) -> Dict[str, Any]:
    """
    Extract a valid JSON triage action from the LLM's raw text response.
    Falls back to a minimal safe action if parsing fails.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", llm_response, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Try to find a JSON object
    json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            # Always ensure email_id is correct
            parsed["email_id"] = parsed.get("email_id", email_id) or email_id
            return parsed
        except json.JSONDecodeError:
            pass

    print(
        f"[DEBUG] Could not parse JSON from LLM response: {llm_response!r}",
        flush=True,
    )
    # Fallback: minimal action that avoids crashing
    return {
        "email_id": email_id,
        "priority": "normal",
        "department": "IT",
        "is_spam": False,
        "response_draft": None,
    }


# Task runner

async def run_task(
    task: str,
    llm_client: OpenAI,
    http_client: httpx.AsyncClient,
) -> float:
    """
    Run a single task episode and return the final normalised score [0.0, 1.0].
    Emits [START], [STEP]…, [END] log lines.
    """
    cfg = TASK_CONFIG[task]
    max_steps: int = cfg["max_steps"]
    max_total_reward: float = cfg["max_total_reward"]

    system_prompt = _build_system_prompt(task)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        result = await env_reset(http_client, task)
        observation = result.get("observation", {})
        last_reward = 0.0

        for step_num in range(1, max_steps + 1):
            if result.get("done", False):
                break

            email = observation.get("current_email")
            if email is None:
                break

            current_email_id: str = email.get("email_id", "E000")

            # LLM decision
            user_prompt = _build_user_prompt(
                observation, step_num, last_reward, history
            )
            raw_response = _call_llm(llm_client, system_prompt, user_prompt, task)
            action_dict = _parse_action(raw_response, current_email_id)

            # Environment step 
            try:
                result = await env_step(http_client, action_dict)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                observation = result.get("observation", {})
                error = None
            except Exception as exc:
                reward = 0.0
                done = False
                error = str(exc)
                print(f"[DEBUG] env_step error at step {step_num}: {exc}", flush=True)

            rewards.append(reward)
            last_reward = reward
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action_dict,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step_num}: email={current_email_id} "
                f"priority={action_dict.get('priority')} "
                f"dept={action_dict.get('department')} "
                f"spam={action_dict.get('is_spam')} "
                f"→ reward {reward:+.4f}"
            )

            if done:
                break

        # Score
        score = (
            sum(rewards) / max_total_reward
            if max_total_reward > 0
            else 0.0
        )
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task}' crashed: {exc}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# Main

async def main() -> None:
    if not HF_TOKEN:
        print(
            "[ERROR] No API key found. Set HF_TOKEN environment variable.",
            flush=True,
        )
        sys.exit(1)

    # Wait for the env server to be ready (up to 60 s)
    print(f"[DEBUG] Connecting to env server at {ENV_URL} ...", flush=True)
    async with httpx.AsyncClient() as probe:
        for attempt in range(30):
            try:
                r = await probe.get(f"{ENV_URL}/health", timeout=5.0)
                if r.status_code == 200:
                    print(f"[DEBUG] Env server ready ({attempt+1} attempts).", flush=True)
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            print("[ERROR] Env server did not become ready in time.", flush=True)
            sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    task_scores: Dict[str, float] = {}

    async with httpx.AsyncClient() as http_client:
        for task in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"[DEBUG] Starting task: {task}", flush=True)
            print(f"{'='*60}", flush=True)
            score = await run_task(task, llm_client, http_client)
            task_scores[task] = score
            time.sleep(1)  # brief pause between tasks

    # Final aggregate summary
    print("\n" + "=" * 60, flush=True)
    print("[SUMMARY] All tasks complete.", flush=True)
    for task, score in task_scores.items():
        status = "✓ PASS" if score >= SUCCESS_SCORE_THRESHOLD else "✗ FAIL"
        print(f"  {task:25s} score={score:.4f}  {status}", flush=True)

    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    print(f"  {'OVERALL':25s} score={overall:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
