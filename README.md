---
title: Email Triage Hub
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - email
  - triage
  - rl
  - agent
---

# 📬 Email Triage Hub — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

A realistic **business email triage environment** for training and evaluating AI agents.
Agents process a stream of realistic corporate emails and must classify priority, route to
the correct department, detect phishing/spam, and draft professional responses — exactly
what a skilled operations assistant does every day.

---

## 🏢 Motivation

Email overload costs organisations billions in lost productivity annually. Training an agent
that can accurately triage email reduces decision fatigue, ensures critical issues are
escalated immediately, and keeps routine work flowing. Unlike toy environments, every email
in this benchmark is modelled on real corporate communication patterns.

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| **Observation** | Current email (id, sender, subject, body, timestamp, attachment flag) + inbox stats |
| **Action** | JSON with `email_id`, `priority`, `department`, `is_spam`, `response_draft` |
| **Reward** | Per-step partial progress (0.0–1.0 cumulative per episode) |
| **Episodes** | One episode = one inbox of 5–10 emails processed sequentially |
| **API** | REST (FastAPI) — `POST /reset`, `POST /step`, `GET /state` |

---

## 📨 Observation Space

```json
{
  "current_email": {
    "email_id": "E001",
    "sender": "devops@techcorp.com",
    "subject": "URGENT: Production Database Down",
    "body": "...",
    "timestamp": "2024-01-15T09:12:00Z",
    "has_attachment": true
  },
  "inbox_summary": {
    "total_emails": 5,
    "processed": 0,
    "remaining": 5,
    "task_name": "priority_sort",
    "task_description": "..."
  },
  "step_feedback": null,
  "task_description": "Classify each email by priority...",
  "requirements": ["priority: one of urgent | normal | low"],
  "available_actions": ["classify_priority"]
}
```

---

## ⚡ Action Space

```json
{
  "email_id": "E001",
  "priority": "urgent",
  "department": "IT",
  "is_spam": false,
  "response_draft": "Thank you for reporting this critical issue. I am immediately escalating to our senior DBA team and will have a status update within 15 minutes."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `email_id` | string | Always | ID of the email being triaged |
| `priority` | `urgent\|normal\|low` | Tasks 1, 2, 3 (non-spam) | Urgency classification |
| `department` | `IT\|HR\|Sales\|Support\|Finance\|Legal\|Management` | Tasks 2, 3 (non-spam) | Routing target |
| `is_spam` | boolean | Task 3 | Spam/phishing flag |
| `response_draft` | string (50–200 words) | Task 3 urgent emails | Professional reply draft |

---

## 🎯 Tasks

### Task 1 — Priority Classification _(Easy)_
- **Emails:** 5 (mix of urgent/normal/low)
- **Required fields:** `priority`
- **Scoring:** 1/5 reward per correct classification → max = **1.0**
- **Max steps:** 10
- **Success threshold:** ≥ 0.60

**Challenge:** Distinguish genuinely urgent emails (production outages, same-day deal deadlines)
from normal and low-priority items.

---

### Task 2 — Priority & Department Routing _(Medium)_
- **Emails:** 8 (all non-spam)
- **Required fields:** `priority` + `department`
- **Scoring:** per email = 0.5 × priority_correct + 0.5 × dept_correct, divided by 8 → max = **1.0**
- **Max steps:** 20
- **Success threshold:** ≥ 0.55

**Challenge:** Correct two-label classification. An HR incident is urgent AND belongs to HR,
not IT. A software invoice is Finance, not IT.

---

### Task 3 — Full Email Triage _(Hard)_
- **Emails:** 10 (3 urgent + 5 normal + 2 spam/phishing)
- **Required fields:** `is_spam` + `priority` + `department` + `response_draft` (urgent)
- **Scoring (per email):**

| Component | Weight | What earns credit |
|---|---|---|
| Priority accuracy | 30% | Correct urgent/normal/low for non-spam |
| Department accuracy | 30% | Correct routing for non-spam |
| Spam detection | 20% | Correct is_spam flag for all emails |
| Response quality | 20% | Acknowledgment + action + professional length (urgent only) |

- **Max steps:** 30
- **Success threshold:** ≥ 0.50

**Challenge:** Agents must identify subtle phishing (spoofed domains, credential harvesting),
correctly route a diverse inbox, AND write coherent professional replies.

---

## 🏆 Reward Function Design

The reward function provides **dense partial progress signals** — agents receive feedback
after every email, not just at episode end.

```
Task 1:  reward_per_step = priority_score / n_emails
Task 2:  reward_per_step = (0.5×priority + 0.5×dept) / n_emails
Task 3:  reward_per_step = priority×0.30/n_ns + dept×0.30/n_ns
                         + spam×0.20/n_total + response×0.20/n_urgent
```

where `n_ns` = non-spam email count, `n_urgent` = urgent-with-response count.

Graders are **fully deterministic**: priority/department/spam use exact-match lookup against
ground-truth labels; response quality uses a reproducible keyword rubric (no LLM judge).

---

## 🚀 Setup & Usage

### Local development

```bash
# Clone and install
git clone <your-repo>
cd email-triage-hub
pip install -r requirements.txt

# Start the environment server
python server.py
# Server at http://localhost:8000

# Run the baseline inference script
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."          # or OPENAI_API_KEY
export ENV_URL="http://localhost:8000"
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-hub .

# Run environment server
docker run -p 8000:8000 email-triage-hub

# Run inference against the running container
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  -e ENV_URL="http://host.docker.internal:8000" \
  email-triage-hub python inference.py
```

### HuggingFace Spaces

1. Push this repository to a HuggingFace Space with the `Docker` SDK.
2. Set `PORT=7860` in the Space secrets.
3. Add `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` as Space secrets.

---

## 🔌 API Reference

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/ping` | — | Alias for /health |
| `GET` | `/tasks` | — | List all tasks |
| `POST` | `/reset` | `{"task": "priority_sort"}` | Start new episode |
| `POST` | `/step` | `TriageAction JSON` | Submit triage decision |
| `GET` | `/state` | — | Inspect full environment state |
| `GET` | `/openenv.yaml` | — | Environment manifest |

### Example curl session

```bash
# Reset for task 1
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "priority_sort"}' | python -m json.tool

# Submit a triage action
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"email_id": "E001", "priority": "urgent"}' | python -m json.tool
```

---

## 📊 Baseline Scores

Baseline agent: **gpt-4o-mini** (temperature=0, zero-shot prompting)

| Task | Score | Pass? |
|---|---|---|
| priority_sort | ~0.80 | ✓ |
| department_routing | ~0.65 | ✓ |
| full_triage | ~0.55 | ✓ |
| **Overall** | **~0.67** | ✓ |

---

## 📁 Project Structure

```
email-triage-hub/
├── server.py              # FastAPI server (OpenEnv API)
├── inference.py           # Baseline LLM inference script
├── openenv.yaml           # Environment manifest
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── env/
    ├── __init__.py
    ├── email_triage_env.py  # Core environment class
    ├── models.py            # Pydantic observation/action/reward models
    ├── tasks.py             # Task definitions & metadata
    ├── graders.py           # Deterministic scoring functions
    └── email_data.py        # 15-email dataset with ground-truth labels
```

---

## 🧪 Testing

```bash
# Quick smoke test – reset and one step
python - <<'EOF'
import httpx, json

base = "http://localhost:8000"
r = httpx.post(f"{base}/reset", json={"task": "priority_sort"})
obs = r.json()
email_id = obs["observation"]["current_email"]["email_id"]
print("First email:", email_id)

r2 = httpx.post(f"{base}/step", json={"email_id": email_id, "priority": "urgent"})
print("Reward:", r2.json()["reward"], "Done:", r2.json()["done"])
EOF
```

---

## License

MIT
