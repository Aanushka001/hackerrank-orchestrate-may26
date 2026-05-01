# Support Triage Agent — Code

## Architecture

```
main.py        ← Terminal entry point; reads support_tickets.csv, writes output.csv
agent.py       ← Core triage logic; calls Claude API with RAG context
retriever.py   ← TF-IDF retriever over local corpus (data/ directory)
```

## Flow

```
support_tickets.csv
        │
        ▼
   retriever.py        ← TF-IDF over data/hackerrank, data/claude, data/visa
        │ top-5 docs
        ▼
    agent.py           ← Claude claude-sonnet-4 with grounded system prompt
        │ JSON output
        ▼
   output.csv
```

## Setup

```bash
# 1. Clone repo and cd into it
git clone git@github.com:interviewstreet/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26

# 2. Install dependencies
pip install anthropic pandas numpy scikit-learn python-dotenv tqdm

# 3. Set API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python code/main.py
```

## Commands

```bash
# Process all tickets (default)
python code/main.py

# Custom input/output paths
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv

# Triage a single issue interactively
python code/main.py --single "My Visa card was charged twice for the same transaction"

# Check corpus stats (no API calls)
python code/main.py --dry-run

# Quiet mode (less output)
python code/main.py --quiet
```

## Design Decisions

### Why TF-IDF (not embeddings)?
- Zero setup — no vector DB, no embedding API calls, no extra cost
- Fast, deterministic, reproducible
- Sufficient for keyword-heavy support queries
- Can swap to sentence-transformers later if needed

### Why claude-sonnet-4?
- Best reasoning for nuanced escalation decisions
- Structured JSON output with temperature=0 for determinism
- Strong grounding — won't hallucinate policies

### Escalation Logic
The agent escalates when:
1. Fraud/security/stolen card issues → always escalate
2. No corpus coverage → escalate + explain gap
3. Sensitive PII or account credentials mentioned
4. Billing disputes needing account access
5. Legal/compliance issues

### Grounding
- All responses cite only the retrieved corpus excerpts
- System prompt explicitly forbids inventing policies
- If corpus is empty for a company, agent acknowledges and escalates

## Output Schema

| Column | Values |
|--------|--------|
| status | `replied` \| `escalated` |
| product_area | free text category |
| response | user-facing answer |
| justification | internal reasoning note |
| request_type | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

## Dependencies

```
anthropic>=0.40.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```