# Multi-Domain Support Triage Agent
### HackerRank Orchestrate — May 2026

RAG-based support triage agent that classifies, retrieves, and responds to support tickets across **HackerRank**, **Claude**, and **Visa** using only the provided local corpus.

**GitHub:** https://github.com/Aanushka001/hackerrank-orchestrate-may26

---

## Architecture

```
support_tickets.csv
        │
        ▼
retriever.py ── TF-IDF over data/
│               ├── HackerRank : 438 docs
│               ├── Claude     : 321 docs
│               └── Visa       :  14 docs
│               top-6 results (score > 0.01)
        │
        ▼
agent.py
├── Stage 1: Keyword pre-escalation (deterministic, no API)
│           stolen card · unauthorized transaction · identity theft
│           account hacked · data breach · security vulnerability
│           phishing · legal action · gdpr request
│
└── Stage 2: Groq LLM (llama-3.1-8b-instant, temperature=0)
            Corpus-grounded system prompt → validated JSON
        │
        ▼
output.csv
status · product_area · response · justification · request_type
```

---

## Features

- **RAG pipeline** — TF-IDF retrieval scoped per company, falls back to full corpus
- **Two-stage escalation** — deterministic keyword pre-check before any LLM call
- **Strict grounding** — system prompt forbids inventing policies or URLs not in corpus
- **Deterministic output** — temperature=0, same input always produces same output
- **Robust JSON parsing** — strips markdown fences, falls back to regex extraction
- **Output validation** — enforces allowed values for `status` and `request_type`
- **No hardcoded secrets** — API key loaded from `.env` only

---

## Setup

```bash
pip install groq scikit-learn python-dotenv tqdm

# Create .env at repo root
echo GROQ_API_KEY=your_key_here > .env

# Get key at: https://console.groq.com
```

---

## How to Run

```bash
# Full batch — all tickets
python code/main.py

# Custom input/output paths
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv

# Single ticket — interactive
python code/main.py --single "My Visa card was stolen"

# Corpus stats — no API call
python code/main.py --dry-run

# Quiet mode
python code/main.py --quiet
```

---

## Input Format

| Field | Description |
|---|---|
| `issue` | Main ticket body (required) |
| `subject` | Optional subject line (may be blank or noisy) |
| `company` | `HackerRank`, `Claude`, `Visa`, or `None` |

---

## Output Format

| Field | Allowed Values |
|---|---|
| `status` | `replied` or `escalated` |
| `product_area` | e.g. Account Access, Fraud Prevention, Card Services, Test Integrity |
| `response` | User-facing answer, corpus-grounded only |
| `justification` | Internal reasoning citing corpus evidence |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

---

## Escalation Logic

**Pre-escalated (no LLM) when issue contains:**
stolen card · lost card · unauthorized transaction · unauthorized charge ·
fraudulent transaction · identity theft · account hacked · account compromised ·
data breach · security vulnerability · phishing · legal action · lawsuit · gdpr request

**LLM escalates when:**
- Corpus has no relevant coverage
- Issue requires PII/credential verification by a human
- Billing dispute with no self-serve option in corpus
- Abusive or malicious content

**LLM replies when:**
- How-to questions covered by corpus
- Feature requests (acknowledged and classified)
- Out-of-scope issues → `status=replied`, `request_type=invalid`
- Third-party decisions outside support scope → explained, not escalated

---

## Logging

Every run appends to `~/hackerrank_orchestrate/log.txt` per AGENTS.md spec.

Log entries include: timestamp, input/output paths, per-run summary (replied/escalated/errors).

---

## Security

- API key read from environment only (`os.environ.get("GROQ_API_KEY")`)
- `.env` is gitignored
- No keys hardcoded anywhere in source

---

## Dependencies

```
groq>=0.9.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

---

## Submission Checklist

- [x] `code/` — agent.py, main.py, retriever.py, README.md
- [x] `support_tickets/output.csv` — 29 tickets processed
- [x] `~/hackerrank_orchestrate/log.txt` — chat transcript
- [x] No hardcoded API keys
- [x] No hallucinated policies — corpus-only grounding
- [x] Deterministic output — temperature=0
- [x] Escalation logic verified on fraud + FAQ tickets
- [x] JSON output validated on every row