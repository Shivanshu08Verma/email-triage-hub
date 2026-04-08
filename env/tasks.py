"""
Task definitions for the Email Triage Hub OpenEnv environment.

Three tasks with increasing difficulty:
  1. priority_sort       – Easy   – classify priority only
  2. department_routing  – Medium – classify priority + route to department
  3. full_triage         – Hard   – spam detection + priority + routing + response drafts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TaskSpec:
    id: str
    name: str
    difficulty: str        # easy | medium | hard
    description: str
    max_steps: int
    max_total_reward: float = 1.0
    requirements: List[str] = field(default_factory=list)
    available_actions: List[str] = field(default_factory=list)
    success_threshold: float = 0.6   # score >= this counts as solved


# Task registry

TASK_REGISTRY: dict[str, TaskSpec] = {
    "priority_sort": TaskSpec(
        id="priority_sort",
        name="Priority Classification",
        difficulty="easy",
        description=(
            "You will receive 5 emails one by one. "
            "For each email, classify its urgency level as one of: urgent, normal, or low.\n\n"
            "  urgent – requires immediate action (system outages, security incidents, "
            "time-sensitive deals with same-day deadlines).\n"
            "  normal – important but can be addressed within a few hours to a day.\n"
            "  low    – informational, optional, or routine; no urgency.\n\n"
            "Submit ONLY the priority field in your JSON action."
        ),
        max_steps=10,
        requirements=["priority: one of urgent | normal | low"],
        available_actions=["classify_priority"],
        success_threshold=0.6,
    ),

    "department_routing": TaskSpec(
        id="department_routing",
        name="Priority & Department Routing",
        difficulty="medium",
        description=(
            "You will receive 8 emails. For each email you must:\n"
            "  1. Classify priority (urgent | normal | low)\n"
            "  2. Route to the correct department\n\n"
            "Valid departments:\n"
            "  IT         – technical issues, systems, infrastructure, security\n"
            "  HR         – people, hiring, onboarding, benefits, incidents\n"
            "  Sales      – leads, deals, renewals, partnerships, pricing\n"
            "  Support    – customer complaints, billing issues, refunds\n"
            "  Finance    – invoices, payments, budgets, procurement\n"
            "  Legal      – contracts, NDAs, compliance, regulatory\n"
            "  Management – executive decisions, company strategy"
        ),
        max_steps=20,
        requirements=[
            "priority: one of urgent | normal | low",
            "department: one of IT | HR | Sales | Support | Finance | Legal | Management",
        ],
        available_actions=["classify_priority_and_route"],
        success_threshold=0.55,
    ),

    "full_triage": TaskSpec(
        id="full_triage",
        name="Full Email Triage",
        difficulty="hard",
        description=(
            "You will receive 10 emails (including spam). For each email:\n\n"
            "  Step 1 – Spam detection\n"
            "    Set is_spam=true for phishing / unsolicited marketing emails.\n"
            "    Spam signals: suspicious sender domains, prize claims, urgent credential requests, "
            "mismatched URLs, grammatical errors designed to deceive.\n\n"
            "  Step 2 – Non-spam emails only\n"
            "    Classify priority (urgent | normal | low) and route to department "
            "(IT | HR | Sales | Support | Finance | Legal | Management).\n\n"
            "  Step 3 – Urgent non-spam emails only\n"
            "    Write a brief professional response_draft (50–200 words) that:\n"
            "      a) Acknowledges the issue with empathy\n"
            "      b) States concrete next action(s) being taken\n"
            "      c) Uses professional business language\n\n"
            "Scoring:\n"
            "  Priority accuracy  30 %\n"
            "  Department accuracy 30 %\n"
            "  Spam detection     20 %\n"
            "  Response quality   20 %"
        ),
        max_steps=30,
        requirements=[
            "is_spam: true | false",
            "priority: urgent | normal | low  (non-spam only)",
            "department: IT | HR | Sales | Support | Finance | Legal | Management  (non-spam only)",
            "response_draft: 50–200 word professional reply  (urgent non-spam only)",
        ],
        available_actions=["mark_spam", "full_triage_non_spam"],
        success_threshold=0.5,
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. Valid tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
