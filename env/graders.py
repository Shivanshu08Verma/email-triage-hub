"""
Deterministic graders for the Email Triage Hub.

Each grader function:
  - Takes an action dict and the email_id (plus any extra context)
  - Returns a float reward in [0.0, 1.0] representing per-step contribution
  - Is fully deterministic (no randomness, no LLM calls)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from env.email_data import EMAIL_MAP

# Constants

VALID_PRIORITIES = {"urgent", "normal", "low"}
VALID_DEPARTMENTS = {"IT", "HR", "Sales", "Support", "Finance", "Legal", "Management"}

#Atomic graders

def _grade_priority(action_priority: Optional[str], email_id: str) -> float:
    """1.0 if priority matches ground truth, 0.0 otherwise."""
    if not action_priority:
        return 0.0
    if action_priority.lower().strip() not in VALID_PRIORITIES:
        return 0.0
    true_priority = EMAIL_MAP.get(email_id, {}).get("true_priority", "")
    return 1.0 if action_priority.lower().strip() == true_priority else 0.0


def _grade_department(action_department: Optional[str], email_id: str) -> float:
    """1.0 if department matches ground truth, 0.0 otherwise."""
    if not action_department:
        return 0.0
    action_dept = action_department.strip()
    if action_dept not in VALID_DEPARTMENTS:
        return 0.0
    true_department = EMAIL_MAP.get(email_id, {}).get("true_department")
    if true_department is None:
        # Spam emails have no expected department
        return 0.0
    return 1.0 if action_dept == true_department else 0.0


def _grade_spam(action_is_spam: Optional[bool], email_id: str) -> float:
    """1.0 if spam detection matches ground truth, 0.0 otherwise."""
    if action_is_spam is None:
        return 0.0
    true_is_spam = EMAIL_MAP.get(email_id, {}).get("is_spam", False)
    return 1.0 if bool(action_is_spam) == true_is_spam else 0.0


def _grade_response(response_draft: Optional[str], email_id: str) -> float:
    """
    Rubric-based response quality grader (deterministic keyword matching).

    Returns 0.0–1.0 based on three equally-weighted criteria:
      1. Acknowledgment / empathy  (0.34)
      2. Concrete action / next steps  (0.33)
      3. Professional length (15–300 words)  (0.33)
    """
    email = EMAIL_MAP.get(email_id, {})

    # Emails that do not require a response get full credit automatically
    if not email.get("needs_response", False):
        return 1.0
    # Only urgent emails in task 3 need a draft; others get full credit
    if email.get("true_priority") != "urgent":
        return 1.0

    if not response_draft or len(response_draft.strip()) < 20:
        return 0.0

    text = response_draft.lower()
    score = 0.0

    # Criterion 1 – Acknowledgment / empathy
    ack_words = [
        "thank", "understand", "acknowledge", "sorry", "received",
        "apolog", "appreciate", "noted", "aware",
    ]
    if any(w in text for w in ack_words):
        score += 0.34

    # Criterion 2 – Concrete action / next steps
    action_words = [
        "will", "immediately", "escalat", "investigat", "contact",
        "team", "fix", "resolv", "address", "follow", "priorit",
        "assign", "looping", "working", "actioning",
    ]
    if any(w in text for w in action_words):
        score += 0.33

    # Criterion 3 – Professional length
    word_count = len(response_draft.split())
    if 15 <= word_count <= 300:
        score += 0.33

    return min(round(score, 4), 1.0)


# Per-task reward functions

def compute_task1_reward(
    action: Dict[str, Any], email_id: str, total_emails: int
) -> float:
    """
    Task 1 – Priority Sort (Easy).

    Perfect priority classification on all emails → cumulative reward = 1.0.
    Each email contributes equally: 1 / total_emails.
    """
    if total_emails <= 0:
        return 0.0
    priority_score = _grade_priority(action.get("priority"), email_id)
    return round(priority_score / total_emails, 6)


def compute_task2_reward(
    action: Dict[str, Any], email_id: str, total_emails: int
) -> float:
    """
    Task 2 – Department Routing (Medium).

    Each email:  50 % priority + 50 % department  →  reward = combined / total_emails
    Perfect on all → cumulative reward = 1.0.
    """
    if total_emails <= 0:
        return 0.0
    priority_score = _grade_priority(action.get("priority"), email_id)
    department_score = _grade_department(action.get("department"), email_id)
    combined = priority_score * 0.5 + department_score * 0.5
    return round(combined / total_emails, 6)


def compute_task3_reward(
    action: Dict[str, Any],
    email_id: str,
    email_ids: List[str],
) -> float:
    """
    Task 3 – Full Triage (Hard).

    Budget allocation across the episode:
      - Priority classification (non-spam): 30 %
      - Department routing (non-spam):      30 %
      - Spam detection (all emails):        20 %
      - Response drafting (urgent non-spam): 20 %

    Each category's budget is split equally among the emails that contribute to it.
    Perfect performance → cumulative reward = 1.0.
    """
    email_data = EMAIL_MAP.get(email_id, {})
    is_spam_email = email_data.get("is_spam", False)

    # Pre-compute counts (deterministic given email_ids list)
    n_total = len(email_ids)
    n_non_spam = sum(
        1 for eid in email_ids if not EMAIL_MAP.get(eid, {}).get("is_spam", False)
    )
    n_spam = n_total - n_non_spam
    n_urgent_need_response = sum(
        1
        for eid in email_ids
        if (
            not EMAIL_MAP.get(eid, {}).get("is_spam", False)
            and EMAIL_MAP.get(eid, {}).get("true_priority") == "urgent"
            and EMAIL_MAP.get(eid, {}).get("needs_response", False)
        )
    )

    # Per-email weight for each component
    w_priority = 0.30 / max(n_non_spam, 1)
    w_department = 0.30 / max(n_non_spam, 1)
    w_spam = 0.20 / max(n_total, 1)
    w_response = 0.20 / max(n_urgent_need_response, 1)

    reward = 0.0

    # Spam detection applies to every email
    reward += _grade_spam(action.get("is_spam"), email_id) * w_spam

    if is_spam_email:
        # Spam emails only contribute spam-detection credit
        return round(reward, 6)

    # Non-spam emails
    reward += _grade_priority(action.get("priority"), email_id) * w_priority
    reward += _grade_department(action.get("department"), email_id) * w_department

    # Response only for urgent emails that require one
    needs_response = (
        email_data.get("true_priority") == "urgent"
        and email_data.get("needs_response", False)
    )
    if needs_response:
        reward += _grade_response(action.get("response_draft"), email_id) * w_response

    return round(reward, 6)


def compute_episode_score(
    processed_actions: list, task: str, email_ids: List[str]
) -> Dict[str, Any]:
    """
    Compute a final breakdown score for the entire episode.
    Returns a dict with per-component accuracies and overall score in [0.0, 1.0].
    """
    from env.email_data import EMAIL_MAP 

    priority_correct = 0
    priority_total = 0
    dept_correct = 0
    dept_total = 0
    spam_correct = 0
    spam_total = 0
    response_scores = []

    for record in processed_actions:
        a = record["action"]
        eid = record["email_id"]
        email_data = EMAIL_MAP.get(eid, {})
        is_spam = email_data.get("is_spam", False)

        spam_total += 1
        if task == "full_triage":
            if _grade_spam(a.get("is_spam"), eid) == 1.0:
                spam_correct += 1

        if not is_spam:
            if task in ("priority_sort", "department_routing", "full_triage"):
                priority_total += 1
                if _grade_priority(a.get("priority"), eid) == 1.0:
                    priority_correct += 1

            if task in ("department_routing", "full_triage"):
                dept_total += 1
                if _grade_department(a.get("department"), eid) == 1.0:
                    dept_correct += 1

            if task == "full_triage" and email_data.get("true_priority") == "urgent" and email_data.get("needs_response"):
                response_scores.append(_grade_response(a.get("response_draft"), eid))

    total_reward = sum(r["reward"] for r in processed_actions)

    return {
        "task": task,
        "total_reward": round(total_reward, 4),
        "priority_accuracy": round(priority_correct / max(priority_total, 1), 4),
        "department_accuracy": round(dept_correct / max(dept_total, 1), 4),
        "spam_accuracy": round(spam_correct / max(spam_total, 1), 4) if task == "full_triage" else None,
        "avg_response_quality": round(sum(response_scores) / max(len(response_scores), 1), 4) if response_scores else None,
        "emails_processed": len(processed_actions),
    }
