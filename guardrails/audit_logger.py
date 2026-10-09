"""
CRRA Lab C4 - Audit Trail

Every agent decision and every human approval is appended to
logs/audit_trail.jsonl, one JSON object per line. The file is opened in append
mode on purpose: re-running the orchestrator adds to the trail, it never
overwrites it. Delete the file by hand if you want a clean demo.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []   # this run only, for the summary

    def log(self, agent: str, action: str, contract_id: str, rationale: str = "",
            approval_status: str = "N/A", actor: str = "system", **extra) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "contract_id": contract_id,
            "rationale": rationale,
            "approval_status": approval_status,
            "actor": actor,
            **extra,
        }
        self.entries.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        short = rationale if len(rationale) <= 80 else rationale[:77] + "..."
        print(f"  [AUDIT] {agent}: {action} ({approval_status}) - {short}")
        return entry

    def summary(self) -> None:
        print(f"\n{'=' * 78}\nAUDIT TRAIL - {len(self.entries)} entries this run\n{'=' * 78}")
        print(f"{'Time':<10}{'Contract':<10}{'Agent':<15}{'Action':<20}{'Approval':<10}Actor")
        print("-" * 78)
        for e in self.entries:
            t = e["timestamp"].split("T")[1]
            print(f"{t:<10}{e['contract_id']:<10}{e['agent']:<15}{e['action'][:19]:<20}"
                  f"{e['approval_status']:<10}{e['actor']}")
        print("-" * 78)
        print(f"Appended to: {self.log_path}")
