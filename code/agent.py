"""
agent.py — Triage agent that classifies, retrieves, and responds to support tickets.
"""
import os
import json
import re
import google.generativeai as genai
from retriever import get_retriever

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"
SYSTEM_PROMPT = """You are a precise support triage agent for a multi-domain help desk covering HackerRank, Claude (AI assistant), and Visa (payment network).

You will receive:
1. A support ticket (issue + subject + company)
2. Retrieved corpus excerpts relevant to the issue

Your job is to produce a JSON object with EXACTLY these fields:
- status: "replied" or "escalated"
- product_area: the most relevant support category (e.g. "Billing", "Account Access", "Test Integrity", "Fraud Prevention", "Card Services", "AI Features", "Integration", etc.)
- response: user-facing response (grounded ONLY in the provided corpus; if escalated, explain why and what will happen next)
- justification: 1-2 sentences explaining your decision and what corpus evidence you used
- request_type: one of "product_issue", "feature_request", "bug", "invalid"

ESCALATION RULES (escalate when ANY of these apply):
- Fraud, unauthorized transactions, or suspected account compromise
- Lost/stolen card or identity theft
- Billing disputes requiring account-level access
- Security vulnerabilities or data breaches
- Legal or compliance issues
- The corpus has NO relevant information to answer the issue
- The issue involves PII, account credentials, or sensitive personal data
- The issue is abusive, harmful, or clearly malicious

REPLY RULES:
- General how-to questions answerable from corpus → replied
- Feature requests with enough context → replied
- Known bugs with workaround documented → replied
- Out-of-scope/irrelevant issues → replied with "out of scope" message

GROUNDING RULE: Never invent policies or steps not present in the corpus excerpts.
If the corpus doesn't cover it, say so and escalate or explain the limitation.

Output ONLY valid JSON. No markdown, no explanation outside JSON."""


def build_user_prompt(issue: str, subject: str, company: str, corpus_chunks: list[dict]) -> str:
    corpus_text = ""
    if corpus_chunks:
        parts = []
        for i, c in enumerate(corpus_chunks, 1):
            parts.append(
                f"[DOC {i} | {c['company']} | {c['filename']} | relevance={c['score']:.2f}]\n{c['text'][:1200]}"
            )
        corpus_text = "\n\n---\n\n".join(parts)
    else:
        corpus_text = "(No relevant corpus documents found for this query.)"

    return f"""SUPPORT TICKET
==============
Company: {company}
Subject: {subject or '(none)'}
Issue: {issue}

RETRIEVED CORPUS EXCERPTS
==========================
{corpus_text}

Produce the JSON triage output now."""


def safe_json_parse(raw: str) -> dict:
    """Extract JSON from model output robustly."""
    # Strip markdown fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    # Fallback
    return {
        "status": "escalated",
        "product_area": "Unknown",
        "response": "Unable to process this ticket automatically. Escalating to human support.",
        "justification": "JSON parsing failed on model output.",
        "request_type": "product_issue",
    }


def validate_output(result: dict) -> dict:
    """Ensure output has required fields with valid values."""
    valid_status = {"replied", "escalated"}
    valid_request_type = {"product_issue", "feature_request", "bug", "invalid"}

    if result.get("status") not in valid_status:
        result["status"] = "escalated"
    if result.get("request_type") not in valid_request_type:
        result["request_type"] = "product_issue"
    for field in ["product_area", "response", "justification"]:
        if not result.get(field):
            result[field] = "N/A"
    return result


def triage_ticket(issue: str, subject: str, company: str) -> dict:
    """Main entry point: triage a single support ticket."""
    # Normalize company
    company_norm = company.strip() if company else "None"

    # Build combined query for retrieval
    query = f"{subject or ''} {issue}".strip()

    # Retrieve relevant docs
    retriever = get_retriever(company_norm if company_norm != "None" else None)
    chunks = retriever.retrieve(query, top_k=5)

    # If company is None and we got nothing, try all companies
    if not chunks and company_norm == "None":
        retriever_all = get_retriever(None)
        chunks = retriever_all.retrieve(query, top_k=5)

    # Build prompts
    user_prompt = build_user_prompt(issue, subject, company_norm, chunks)

    # Call Gemini
    try:
        model_client = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=1000,
            ),
        )
        response = model_client.generate_content(user_prompt)
        raw = response.text or ""
    except Exception as e:
        return {
            "status": "escalated",
            "product_area": "Unknown",
            "response": f"Agent error: {str(e)[:100]}. Escalating to human support.",
            "justification": f"API error during triage: {str(e)[:100]}",
            "request_type": "product_issue",
        }

    result = safe_json_parse(raw)
    result = validate_output(result)
    return result