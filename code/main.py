#!/usr/bin/env python3
"""
main.py — Terminal entry point for the Multi-Domain Support Triage Agent.

Usage:
    python main.py                                     # process all tickets
    python main.py --input path/to/tickets.csv        # custom input
    python main.py --output path/to/output.csv        # custom output
    python main.py --single "My card was stolen"      # triage single issue interactively
    python main.py --dry-run                          # show retriever stats only
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Validate API key early
if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set.")
    if ENV_FILE.exists():
        print("Your .env likely has invalid format. It must be:")
        print("GEMINI_API_KEY=AIza...")
    else:
        print("Copy .env.example -> .env and add your key.")
    sys.exit(1)

INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"

# Log file (per AGENTS.md spec)
import platform
LOG_DIR = Path.home() / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"


def ensure_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def append_log(entry: str):
    ensure_log()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def log_run_start(input_path: str, output_path: str):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    append_log(f"\n## [{ts}] AGENT RUN START\n"
               f"Input: {input_path}\nOutput: {output_path}\n"
               f"tool=main.py\n")
def read_tickets(path: Path) -> list[dict]:
    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.strip().lower(): v.strip() for k, v in row.items()}
            tickets.append(normalized)
    return tickets

def write_output(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["issue", "subject", "company", "status", "product_area", "response", "justification", "request_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_tickets(input_path: Path, output_path: Path, verbose: bool = True):
    from agent import triage_ticket
    from tqdm import tqdm

    tickets = read_tickets(input_path)
    print(f"\nLoaded {len(tickets)} tickets from {input_path}")

    results = []
    errors = 0

    for i, ticket in enumerate(tqdm(tickets, desc="Triaging", unit="ticket"), 1):
        issue = ticket.get("issue", ticket.get("issue_body", ticket.get("ticket", ""))).strip()
        subject = ticket.get("subject", ticket.get("title", "")).strip()
        company = ticket.get("company", ticket.get("domain", "None")).strip() or "None"


        if not issue:
            result = {
                "status": "escalated",
                "product_area": "Unknown",
                "response": "No issue text provided. Escalating for human review.",
                "justification": "Empty issue field.",
                "request_type": "invalid",
            }
        else:
            try:
                result = triage_ticket(issue, subject, company)
            except Exception as e:
                errors += 1
                result = {
                    "status": "escalated",
                    "product_area": "Unknown",
                    "response": "System error during triage. Escalating to human support.",
                    "justification": f"Unhandled exception: {str(e)[:80]}",
                    "request_type": "product_issue",
                }

        row = {**ticket, **result}
        results.append(row)

        if verbose and i % 10 == 0:
            replied = sum(1 for r in results if r.get("status") == "replied")
            escalated = len(results) - replied
            tqdm.write(f"  -> {i}/{len(tickets)} | replied={replied} escalated={escalated}")

        # Small delay to avoid rate limits
        time.sleep(0.3)

    write_output(results, output_path)

    replied = sum(1 for r in results if r.get("status") == "replied")
    escalated = len(results) - replied
    print(f"\nDone! {len(results)} tickets processed.")
    print(f"   replied={replied}  escalated={escalated}  errors={errors}")
    print(f"   Output -> {output_path}")
    append_log(f"Run complete: {len(results)} tickets, replied={replied}, escalated={escalated}, errors={errors}\n")


def single_mode(issue: str):
    """Interactive triage of a single issue."""
    from agent import triage_ticket
    import json

    print(f"\nTriaging: {issue[:80]}...")
    company = input("Company (HackerRank / Claude / Visa / None) [None]: ").strip() or "None"
    subject = input("Subject (optional): ").strip()

    result = triage_ticket(issue, subject, company)
    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
    print("="*60)


def dry_run_mode():
    """Show corpus stats without running any tickets."""
    from retriever import get_retriever
    print("\nCorpus stats:")
    for company in ["HackerRank", "Claude", "Visa"]:
        r = get_retriever(company)
        print(f"  {company}: {r.corpus_size()} documents")
    r_all = get_retriever(None)
    print(f"  ALL: {r_all.corpus_size()} documents total")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=str(INPUT_CSV), help="Input CSV path")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--single", metavar="ISSUE", help="Triage a single issue interactively")
    parser.add_argument("--dry-run", action="store_true", help="Show corpus stats only")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_mode()
        return

    if args.single:
        single_mode(args.single)
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    log_run_start(str(input_path), str(output_path))
    process_tickets(input_path, output_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()