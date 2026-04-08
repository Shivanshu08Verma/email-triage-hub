"""
tests/test_environment.py – Pytest suite for the Email Triage Hub.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import pytest

from env.email_data import EMAIL_MAP, TASK_EMAILS
from env.email_triage_env import EmailTriageEnv
from env.graders import (
    _grade_department,
    _grade_priority,
    _grade_response,
    _grade_spam,
    compute_episode_score,
)
from env.models import EnvironmentState, TriageAction
from env.tasks import TASK_REGISTRY, get_task


# Fixtures 

@pytest.fixture
def env_task1():
    e = EmailTriageEnv(task="priority_sort")
    e.reset()
    return e


@pytest.fixture
def env_task2():
    e = EmailTriageEnv(task="department_routing")
    e.reset()
    return e


@pytest.fixture
def env_task3():
    e = EmailTriageEnv(task="full_triage")
    e.reset()
    return e


RESPONSE_DRAFT = (
    "Thank you for alerting us. I acknowledge the urgency and am immediately "
    "escalating to the senior team. We will investigate and resolve this issue "
    "right away and provide an update within 15 minutes."
)


# Task registry

def test_task_registry_has_three_tasks():
    assert len(TASK_REGISTRY) == 3


def test_task_difficulty_progression():
    difficulties = [spec.difficulty for spec in TASK_REGISTRY.values()]
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


def test_get_task_raises_for_unknown():
    with pytest.raises(ValueError):
        get_task("nonexistent_task")


def test_task_specs_have_required_fields():
    for tid, spec in TASK_REGISTRY.items():
        assert spec.id == tid
        assert spec.max_steps > 0
        assert 0.0 <= spec.success_threshold <= 1.0
        assert len(spec.requirements) > 0


# Atomic graders

@pytest.mark.parametrize("email_id,expected", [
    ("E001", "urgent"),
    ("E002", "urgent"),
    ("E005", "normal"),
    ("E011", "low"),
    ("E013", "low"),
])
def test_grade_priority_correct(email_id, expected):
    assert _grade_priority(expected, email_id) == 1.0


@pytest.mark.parametrize("email_id,wrong", [
    ("E001", "low"),
    ("E005", "urgent"),
])
def test_grade_priority_wrong(email_id, wrong):
    assert _grade_priority(wrong, email_id) == 0.0


def test_grade_priority_none():
    assert _grade_priority(None, "E001") == 0.0


def test_grade_priority_invalid():
    assert _grade_priority("super-urgent", "E001") == 0.0


@pytest.mark.parametrize("email_id,expected_dept", [
    ("E001", "IT"),
    ("E003", "HR"),
    ("E002", "Sales"),
    ("E005", "Finance"),
    ("E010", "Legal"),
])
def test_grade_department_correct(email_id, expected_dept):
    assert _grade_department(expected_dept, email_id) == 1.0


def test_grade_department_wrong():
    assert _grade_department("Finance", "E001") == 0.0


def test_grade_spam_correct_spam():
    assert _grade_spam(True, "E014") == 1.0
    assert _grade_spam(True, "E015") == 1.0


def test_grade_spam_correct_not_spam():
    assert _grade_spam(False, "E001") == 1.0
    assert _grade_spam(False, "E005") == 1.0


def test_grade_spam_incorrect():
    assert _grade_spam(False, "E014") == 0.0
    assert _grade_spam(True, "E001") == 0.0


def test_grade_spam_none():
    assert _grade_spam(None, "E014") == 0.0


def test_grade_response_good_draft():
    score = _grade_response(RESPONSE_DRAFT, "E001")
    assert score > 0.5


def test_grade_response_empty():
    assert _grade_response("", "E001") == 0.0
    assert _grade_response(None, "E001") == 0.0


def test_grade_response_too_short():
    assert _grade_response("ok", "E001") == 0.0


def test_grade_response_non_urgent_gets_full_credit():
    # Non-urgent emails don't need a response draft
    assert _grade_response(None, "E005") == 1.0
    assert _grade_response("", "E005") == 1.0


# Task 1

def test_task1_reset_returns_first_email(env_task1):
    env = EmailTriageEnv(task="priority_sort")
    obs = env.reset()
    assert obs.current_email is not None
    assert obs.current_email.email_id == "E001"
    assert obs.inbox_summary.remaining == 5


def test_task1_perfect_score():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    total = 0.0
    for eid in TASK_EMAILS["priority_sort"]:
        _, r, _, _ = env.step(TriageAction(
            email_id=eid, priority=EMAIL_MAP[eid]["true_priority"]
        ))
        total += r
    assert abs(total - 1.0) < 0.001


def test_task1_all_wrong_score():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    total = 0.0
    for eid in TASK_EMAILS["priority_sort"]:
        _, r, _, _ = env.step(TriageAction(email_id=eid, priority="low"))
        total += r
    # "low" is only correct for E011 and E013 (2/5)
    assert total < 1.0
    assert total >= 0.0


def test_task1_reward_range(env_task1):
    env = env_task1
    for eid in TASK_EMAILS["priority_sort"]:
        _, r, done, _ = env.step(TriageAction(email_id=eid, priority="normal"))
        assert 0.0 <= r <= 1.0


def test_task1_done_after_all_emails():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    done = False
    for eid in TASK_EMAILS["priority_sort"]:
        _, _, done, _ = env.step(TriageAction(email_id=eid, priority="normal"))
    assert done is True


def test_task1_episode_summary_on_done():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    info = {}
    for eid in TASK_EMAILS["priority_sort"]:
        _, _, done, info = env.step(TriageAction(
            email_id=eid, priority=EMAIL_MAP[eid]["true_priority"]
        ))
    assert "episode_summary" in info
    assert info["episode_summary"]["priority_accuracy"] == 1.0


def test_task1_step_after_done_is_harmless():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    for eid in TASK_EMAILS["priority_sort"]:
        env.step(TriageAction(email_id=eid, priority="normal"))
    # Extra step after done
    _, r, done, info = env.step(TriageAction(email_id="E001", priority="urgent"))
    assert r == 0.0
    assert done is True
    assert info.get("message") == "episode_already_done"


def test_task1_email_id_mismatch_warning():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    _, r, _, info = env.step(TriageAction(email_id="WRONG_ID", priority="urgent"))
    assert "warning" in info
    assert r > 0.0  # E001 is urgent, so correct answer still earns reward


def test_task1_reset_clears_state():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    env.step(TriageAction(email_id="E001", priority="urgent"))
    env.reset()
    state = env.state()
    assert state.current_email_index == 0
    assert state.cumulative_reward == 0.0
    assert state.done is False


#Task 2

def test_task2_perfect_score():
    env = EmailTriageEnv(task="department_routing")
    env.reset()
    total = 0.0
    for eid in TASK_EMAILS["department_routing"]:
        ed = EMAIL_MAP[eid]
        _, r, _, _ = env.step(TriageAction(
            email_id=eid,
            priority=ed["true_priority"],
            department=ed["true_department"],
        ))
        total += r
    assert abs(total - 1.0) < 0.001


def test_task2_half_credit_right_priority_wrong_dept():
    env = EmailTriageEnv(task="department_routing")
    env.reset()
    total = 0.0
    for eid in TASK_EMAILS["department_routing"]:
        ed = EMAIL_MAP[eid]
        _, r, _, _ = env.step(TriageAction(
            email_id=eid,
            priority=ed["true_priority"],
            department="Management",  # wrong for most
        ))
        total += r
    # Should be between 0 and 1 but less than perfect
    assert 0.0 < total < 1.0


def test_task2_no_department_gives_partial():
    env = EmailTriageEnv(task="department_routing")
    env.reset()
    ed = EMAIL_MAP["E001"]
    _, r, _, _ = env.step(TriageAction(
        email_id="E001",
        priority=ed["true_priority"],
        department=None,
    ))
    # 50% credit for correct priority
    assert r > 0.0


def test_task2_8_emails():
    env = EmailTriageEnv(task="department_routing")
    obs = env.reset()
    assert obs.inbox_summary.total_emails == 8


# Task 3

def test_task3_perfect_score():
    env = EmailTriageEnv(task="full_triage")
    env.reset()
    total = 0.0
    for eid in TASK_EMAILS["full_triage"]:
        ed = EMAIL_MAP[eid]
        draft = None
        if (not ed["is_spam"] and ed.get("true_priority") == "urgent"
                and ed.get("needs_response")):
            draft = RESPONSE_DRAFT
        _, r, _, _ = env.step(TriageAction(
            email_id=eid,
            priority=ed["true_priority"] if not ed["is_spam"] else None,
            department=ed["true_department"] if not ed["is_spam"] else None,
            is_spam=ed["is_spam"],
            response_draft=draft,
        ))
        total += r
    assert total >= 0.95


def test_task3_spam_detection():
    env = EmailTriageEnv(task="full_triage")
    env.reset()
    # Get to E014 (index 8)
    for eid in TASK_EMAILS["full_triage"][:8]:
        ed = EMAIL_MAP[eid]
        env.step(TriageAction(email_id=eid, is_spam=ed["is_spam"],
                              priority=ed.get("true_priority"),
                              department=ed.get("true_department")))
    # E014 is spam - correct flag
    _, r_correct, _, _ = env.step(TriageAction(email_id="E014", is_spam=True))
    assert r_correct > 0.0


def test_task3_spam_wrong_flag():
    env = EmailTriageEnv(task="full_triage")
    env.reset()
    for eid in TASK_EMAILS["full_triage"][:8]:
        ed = EMAIL_MAP[eid]
        env.step(TriageAction(email_id=eid, is_spam=ed["is_spam"],
                              priority=ed.get("true_priority"),
                              department=ed.get("true_department")))
    _, r_wrong, _, _ = env.step(TriageAction(email_id="E014", is_spam=False))
    assert r_wrong == 0.0  # spam detection wrong → 0 for that component only


def test_task3_10_emails():
    env = EmailTriageEnv(task="full_triage")
    obs = env.reset()
    assert obs.inbox_summary.total_emails == 10


def test_task3_episode_summary_components():
    env = EmailTriageEnv(task="full_triage")
    env.reset()
    info = {}
    for eid in TASK_EMAILS["full_triage"]:
        ed = EMAIL_MAP[eid]
        draft = None
        if (not ed["is_spam"] and ed.get("true_priority") == "urgent"
                and ed.get("needs_response")):
            draft = RESPONSE_DRAFT
        _, _, done, info = env.step(TriageAction(
            email_id=eid,
            priority=ed["true_priority"] if not ed["is_spam"] else None,
            department=ed["true_department"] if not ed["is_spam"] else None,
            is_spam=ed["is_spam"],
            response_draft=draft,
        ))
    summary = info["episode_summary"]
    assert summary["spam_accuracy"] is not None
    assert summary["avg_response_quality"] is not None
    assert summary["priority_accuracy"] == 1.0
    assert summary["department_accuracy"] == 1.0


# State

def test_state_initial():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    s = env.state()
    assert isinstance(s, EnvironmentState)
    assert s.current_email_index == 0
    assert s.cumulative_reward == 0.0
    assert s.done is False
    assert s.task == "priority_sort"


def test_state_after_one_step():
    env = EmailTriageEnv(task="priority_sort")
    env.reset()
    env.step(TriageAction(email_id="E001", priority="urgent"))
    s = env.state()
    assert s.current_email_index == 1
    assert len(s.processed_actions) == 1
    assert s.cumulative_reward > 0.0


# Email dataset

def test_email_data_has_ground_truth():
    for email in EMAIL_MAP.values():
        assert "true_priority" in email
        assert email["true_priority"] in {"urgent", "normal", "low"}
        if not email["is_spam"]:
            assert "true_department" in email
            assert email["true_department"] in {
                "IT", "HR", "Sales", "Support", "Finance", "Legal", "Management"
            }


def test_task_email_ids_exist():
    for task, ids in TASK_EMAILS.items():
        assert len(ids) >= 3
        for eid in ids:
            assert eid in EMAIL_MAP, f"Email {eid} in task {task} missing from EMAIL_MAP"


def test_task1_has_5_emails():
    assert len(TASK_EMAILS["priority_sort"]) == 5


def test_task2_has_8_emails():
    assert len(TASK_EMAILS["department_routing"]) == 8


def test_task3_has_10_emails():
    assert len(TASK_EMAILS["full_triage"]) == 10


def test_task3_has_spam():
    spam_in_task3 = [
        eid for eid in TASK_EMAILS["full_triage"]
        if EMAIL_MAP[eid]["is_spam"]
    ]
    assert len(spam_in_task3) >= 2


def test_task3_has_urgent():
    urgent_in_task3 = [
        eid for eid in TASK_EMAILS["full_triage"]
        if not EMAIL_MAP[eid]["is_spam"]
        and EMAIL_MAP[eid]["true_priority"] == "urgent"
    ]
    assert len(urgent_in_task3) >= 2
