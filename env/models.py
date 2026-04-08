"""
Typed Pydantic models for the Email Triage Hub OpenEnv environment.

Observation  → what the agent sees
Action       → what the agent submits
Reward       → per-step scoring metadata
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Core email data 

class EmailItem(BaseModel):
    """A single email presented to the agent."""
    email_id: str = Field(..., description="Unique identifier, e.g. 'E001'")
    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    timestamp: str = Field(..., description="ISO-8601 send timestamp")
    has_attachment: bool = Field(False, description="Whether attachments are present")


class InboxSummary(BaseModel):
    """High-level summary of the current inbox state."""
    total_emails: int = Field(..., description="Total emails in this episode")
    processed: int = Field(..., description="Emails already triaged")
    remaining: int = Field(..., description="Emails still to process")
    task_name: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Human-readable task description")


# Observation 

class EmailObservation(BaseModel):
    """
    Observation returned by reset() and step().
    None values indicate the episode has ended (no more emails).
    """
    current_email: Optional[EmailItem] = Field(
        None, description="The email the agent must triage next; None when done"
    )
    inbox_summary: InboxSummary = Field(..., description="Inbox / progress statistics")
    step_feedback: Optional[str] = Field(
        None,
        description="Feedback from the previous step (correct/wrong indicators)",
    )
    task_description: str = Field(..., description="What the agent must accomplish")
    requirements: List[str] = Field(
        ..., description="Required fields in the triage action for this task"
    )
    available_actions: List[str] = Field(
        ..., description="Human-readable list of valid action types"
    )


# Action

class TriageAction(BaseModel):
    """
    The agent's triage decision for one email.

    Task 1 (priority_sort):        email_id + priority
    Task 2 (department_routing):   email_id + priority + department
    Task 3 (full_triage):          email_id + is_spam + priority + department + response_draft (urgent only)
    """
    email_id: str = Field(
        ..., description="ID of the email being triaged, e.g. 'E001'"
    )
    priority: Optional[str] = Field(
        None,
        description="Priority level: 'urgent' | 'normal' | 'low'. Required for non-spam emails.",
    )
    department: Optional[str] = Field(
        None,
        description=(
            "Routing department: 'IT' | 'HR' | 'Sales' | 'Support' | 'Finance' | 'Legal' | 'Management'. "
            "Required for tasks 2 and 3 (non-spam)."
        ),
    )
    is_spam: Optional[bool] = Field(
        None, description="True if the email is spam / phishing. Required for task 3."
    )
    response_draft: Optional[str] = Field(
        None,
        description=(
            "Brief professional response draft (50–200 words). "
            "Required for task 3 urgent non-spam emails."
        ),
    )


# API request / response wrappers

class ResetRequest(BaseModel):
    """Body sent to POST /reset."""
    task: str = Field(
        "priority_sort",
        description="Task to start: 'priority_sort' | 'department_routing' | 'full_triage'",
    )


class StepResult(BaseModel):
    """Response from POST /reset and POST /step."""
    observation: EmailObservation
    reward: float = Field(..., description="Per-step reward in [0.0, 1.0]")
    done: bool = Field(..., description="True when the episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metadata")


class EnvironmentState(BaseModel):
    """Full internal state returned by GET /state."""
    task: str
    current_email_index: int
    total_emails: int
    processed_actions: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
