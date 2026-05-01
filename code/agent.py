import os
import json
import re
from groq import Groq
from retriever import get_retriever

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

ESCALATION_KEYWORDS = [
    "stolen card", "card stolen", "lost card", "card lost",
    "block my card", "cancel my card", "replace my card",
    "unauthorized transaction", "unauthorized charge", "unauthorized access",
    "fraudulent transaction", "fraudulent charge",
    "identity theft", "identity stolen",
    "account hacked", "account compromised", "account breached",
    "suspicious login", "suspicious activity",
    "data breach", "security vulnerability", "security exploit",
    "phishing", "scam email",
    "legal action", "lawsuit", "gdpr request", "compliance issue",
    "social security number", "passport number",
]

KEYWORD_PRODUCT_AREA = {
    "stolen card": "Card Security",
    "card stolen": "Card Security",
    "lost card": "Card Services",
    "card lost": "Card Services",
    "unauthorized transaction": "Fraud Prevention",
    "unauthorized charge": "Fraud Prevention",
    "fraudulent transaction": "Fraud Prevention",
    "fraudulent charge": "Fraud Prevention",
    "unauthorized access": "Account Security",
    "identity theft": "Identity Protection",
    "identity stolen": "Identity Protection",
    "account hacked": "Account Security",
    "account compromised": "Account Security",
    "data breach": "Security",
    "security vulnerability": "Security",
    "security exploit": "Security",
    "phishing": "Security",
    "legal action": "Legal & Compliance",
    "lawsuit": "Legal & Compliance",
    "gdpr request": "Legal & Compliance",
}


def pre_escalation_check(issue: str, subject: str) -> dict | None:
    combined = f"{subject} {issue}".lower()
    matched = None
    for kw in ESCALATION_KEYWORDS:
        if kw in combined:
            matched = kw
            break
    if not matched:
        return None

    area = KEYWORD_PRODUCT_AREA.get(matched, "Security & Fraud")
    return {
        "status": "escalated",
        "product_area": area,
        "response": (
            "Your case has been escalated to our specialized support team. "
            "A human agent will contact you as soon as possible. "
            "If this involves a lost or stolen card, please also contact your card issuer "
            "directly to block it immediately. Do not share sensitive account details here."
        ),
        "justification": (
            f"Pre-escalated: high-risk keyword '{matched}' detected. "
            "Fraud, security, and card theft issues require human agent handling."
        ),
        "request_type": "product_issue",
    }


SYSTEM_PROMPT = """You are a precise support triage agent for a multi-domain help desk covering HackerRank, Claude (AI assistant), and Visa (payment network).

You receive a support ticket and retrieved corpus excerpts. Produce a JSON object with EXACTLY these fields:
- status: "replied" or "escalated"
- product_area: the most relevant support category (e.g. "Billing", "Account Access", "Test Integrity", "Developer Tools", "Card Services", "AI Features", "Integration", "Subscription", etc.)
- response: user-facing response grounded ONLY in the provided corpus excerpts
- justification: 1-2 sentences explaining your decision and which corpus evidence you used
- request_type: one of "product_issue", "feature_request", "bug", "invalid"

ESCALATION RULES — set status=escalated when:
- Corpus has NO relevant information to answer the issue
- Issue involves PII or credentials requiring human verification
- Issue is abusive or clearly malicious
- Billing dispute requiring account-level access with no self-serve option in corpus

REPLY RULES — set status=replied when:
- How-to questions answerable from corpus
- Feature requests (acknowledge and classify)
- Known bugs with workaround in corpus
- Out-of-scope or irrelevant issues: reply politely, set request_type=invalid

GROUNDING: Use only the provided corpus excerpts. Never invent policies, URLs, or steps. If corpus is insufficient, escalate and explain the gap.

Output ONLY valid JSON. No markdown, no text outside the JSON."""


def build_user_prompt(issue: str, subject: str, company: str, chunks: list[dict]) -> str:
    if chunks:
        parts = [
            f"[DOC {i} | {c['company']} | {c['filename']} | score={c['score']:.3f}]\n{c['text'][:1500]}"
            for i, c in enumerate(chunks, 1)
        ]
        corpus_text = "\n\n---\n\n".join(parts)
    else:
        corpus_text = "(No relevant corpus documents found.)"

    return f"""SUPPORT TICKET
==============
Company: {company}
Subject: {subject or '(none)'}
Issue: {issue}

RETRIEVED CORPUS EXCERPTS
==========================
{corpus_text}

Output ONLY valid JSON now."""


def safe_json_parse(raw: str) -> dict:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {
        "status": "escalated",
        "product_area": "Unknown",
        "response": "Unable to process this ticket automatically. Escalating to human support.",
        "justification": "JSON parsing failed on model output.",
        "request_type": "product_issue",
    }


def validate_output(result: dict) -> dict:
    if result.get("status") not in {"replied", "escalated"}:
        result["status"] = "escalated"
    if result.get("request_type") not in {"product_issue", "feature_request", "bug", "invalid"}:
        result["request_type"] = "product_issue"
    for field in ["product_area", "response", "justification"]:
        if not result.get(field) or str(result[field]).strip() in ("", "N/A", "null"):
            result[field] = "Not available"
    return result


def triage_ticket(issue: str, subject: str, company: str) -> dict:
    company_norm = (company or "None").strip()

    pre = pre_escalation_check(issue, subject)
    if pre:
        return pre

    query = f"{subject or ''} {issue}".strip()

    if company_norm != "None":
        chunks = get_retriever(company_norm).retrieve(query, top_k=6)
        if not chunks:
            chunks = get_retriever(None).retrieve(query, top_k=6)
    else:
        chunks = get_retriever(None).retrieve(query, top_k=6)

    chunks = [c for c in chunks if c["score"] > 0.01]

    user_prompt = build_user_prompt(issue, subject, company_norm, chunks)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        return {
            "status": "escalated",
            "product_area": "Unknown",
            "response": "An error occurred during triage. Escalating to human support.",
            "justification": f"API error: {str(e)[:120]}",
            "request_type": "product_issue",
        }

    return validate_output(safe_json_parse(raw))