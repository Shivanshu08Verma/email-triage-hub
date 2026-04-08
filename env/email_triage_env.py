"""
EmailTriageEnv – Core environment class for the Email Triage Hub.

Implements the OpenEnv interface:
  reset()  → EmailObservation
  step()   → (EmailObservation, reward, done, info)
  state()  → EnvironmentState

State is kept in-memory; one active episode per instance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.email_data import EMAIL_MAP, TASK_EMAILS
from env.graders import (
    compute_task1_reward,
    compute_task2_reward,
    compute_task3_reward,
    compute_episode_score,
)
from env.models import (
    EmailItem,
    EmailObservation,
    EnvironmentState,
    InboxSummary,
    TriageAction,
)
from env.tasks import TASK_REGISTRY, get_task


class EmailTriageEnv:
    """
    Stateful email triage environment.

    Episodes:
      - reset(task) initialises a fresh episode.
      - step(action) processes one email at a time and returns per-step reward.
      - state() returns full internal state for debugging / inspection.
    """

    def __init__(self, task: str = "priority_sort") -> None:
        self._task_spec = get_task(task)
        self.task: str = task
        self.email_ids: List[str] = TASK_EMAILS[task]
        self.current_index: int = 0
        self.processed_actions: List[Dict[str, Any]] = []
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self._last_feedback: Optional[str] = None

    # Public interface 

    def reset(self) -> EmailObservation:
        """Reset the environment for a new episode. Returns the first observation."""
        self.current_index = 0
        self.processed_actions = []
        self.cumulative_reward = 0.0
        self.done = False
        self._last_feedback = None
        return self._make_observation()

    def step(
        self, action: TriageAction
    ) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:
        """
        Process one triage action.

        Returns:
            observation  – next email (or None if done)
            reward       – per-step reward in [0.0, task_max_per_step]
            done         – True when all emails are processed
            info         – auxiliary metadata dict
        """
        # Guard: already finished
        if self.done:
            obs = self._make_observation(
                feedback="Episode finished. Call /reset to start a new episode."
            )
            return obs, 0.0, True, {"message": "episode_already_done"}

        # Guard: no more emails (should not normally occur)
        if self.current_index >= len(self.email_ids):
            self.done = True
            obs = self._make_observation(feedback="All emails processed.")
            return obs, 0.0, True, {"message": "all_emails_processed"}

        current_email_id = self.email_ids[self.current_index]

        # Validate email_id in action matches the current email.
        # If mismatched (common LLM mistake), override with the correct id
        # but record a warning. This keeps episodes from stalling.
        email_id_warning: Optional[str] = None
        if action.email_id and action.email_id != current_email_id:
            email_id_warning = (
                f"email_id mismatch: submitted '{action.email_id}' "
                f"but current email is '{current_email_id}'. "
                f"Graded against '{current_email_id}'."
            )
            action = action.model_copy(update={"email_id": current_email_id})

        # Compute reward (action.email_id is now guaranteed == current_email_id)
        reward = self._compute_reward(action, current_email_id)

        # Build step-level feedback
        feedback = self._build_feedback(action, current_email_id, reward)
        self._last_feedback = feedback

        # Record
        self.processed_actions.append(
            {
                "email_id": current_email_id,
                "action": action.model_dump(),
                "reward": reward,
                "step": self.current_index + 1,
            }
        )
        self.cumulative_reward = round(self.cumulative_reward + reward, 6)

        # Advance
        self.current_index += 1
        self.done = self.current_index >= len(self.email_ids)

        obs = self._make_observation()
        info: Dict[str, Any] = {
            "triaged_email_id": current_email_id,
            "step_reward": round(reward, 6),
            "cumulative_reward": self.cumulative_reward,
            "emails_remaining": max(0, len(self.email_ids) - self.current_index),
            "done": self.done,
        }
        if email_id_warning:
            info["warning"] = email_id_warning

        if self.done:
            summary = compute_episode_score(
                self.processed_actions, self.task, self.email_ids
            )
            info["episode_summary"] = summary

        return obs, reward, self.done, info

    def state(self) -> EnvironmentState:
        """Return full internal state."""
        return EnvironmentState(
            task=self.task,
            current_email_index=self.current_index,
            total_emails=len(self.email_ids),
            processed_actions=self.processed_actions,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
        )

    # Private helpers

    def _compute_reward(self, action: TriageAction, email_id: str) -> float:
        action_dict = action.model_dump()
        if self.task == "priority_sort":
            return compute_task1_reward(action_dict, email_id, len(self.email_ids))
        elif self.task == "department_routing":
            return compute_task2_reward(action_dict, email_id, len(self.email_ids))
        elif self.task == "full_triage":
            return compute_task3_reward(action_dict, email_id, self.email_ids)
        return 0.0

    def _build_feedback(
        self, action: TriageAction, email_id: str, reward: float
    ) -> str:
        """Construct human-readable step feedback."""
        email_data = EMAIL_MAP.get(email_id, {})
        true_priority = email_data.get("true_priority", "?")
        true_dept = email_data.get("true_department", "?")
        is_spam = email_data.get("is_spam", False)

        parts = [
            f"[Email {email_id}] Step reward: {reward:.4f} | "
            f"Cumulative: {self.cumulative_reward + reward:.4f}"
        ]

        if self.task in ("priority_sort", "department_routing", "full_triage"):
            if not is_spam:
                p_ok = action.priority and action.priority.lower() == true_priority
                parts.append(
                    f"Priority {'✓' if p_ok else '✗'} "
                    f"(submitted: {action.priority!r}, correct: {true_priority!r})"
                )

        if self.task in ("department_routing", "full_triage"):
            if not is_spam:
                d_ok = action.department == true_dept
                parts.append(
                    f"Department {'✓' if d_ok else '✗'} "
                    f"(submitted: {action.department!r}, correct: {true_dept!r})"
                )

        if self.task == "full_triage":
            s_ok = action.is_spam == is_spam
            parts.append(
                f"Spam {'✓' if s_ok else '✗'} "
                f"(submitted: {action.is_spam}, correct: {is_spam})"
            )

        return " | ".join(parts)

    def _make_observation(
        self, feedback: Optional[str] = None
    ) -> EmailObservation:
        """Build the observation for the current state."""
        current_email: Optional[EmailItem] = None

        if self.current_index < len(self.email_ids):
            raw = EMAIL_MAP[self.email_ids[self.current_index]]
            current_email = EmailItem(
                email_id=raw["email_id"],
                sender=raw["sender"],
                subject=raw["subject"],
                body=raw["body"],
                timestamp=raw["timestamp"],
                has_attachment=raw["has_attachment"],
            )

        spec = self._task_spec
        return EmailObservation(
            current_email=current_email,
            inbox_summary=InboxSummary(
                total_emails=len(self.email_ids),
                processed=self.current_index,
                remaining=max(0, len(self.email_ids) - self.current_index),
                task_name=self.task,
                task_description=spec.description,
            ),
            step_feedback=feedback or self._last_feedback,
            task_description=spec.description,
            requirements=list(spec.requirements),
            available_actions=list(spec.available_actions),
        )
