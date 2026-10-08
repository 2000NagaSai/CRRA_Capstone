"""
CRRA Lab C2 - Mock Contract Management API

Serves the contract portfolio in data/contracts.csv with derived policy fields
(notice_deadline, notice_state, utilisation_pct, approval_band), computed once at
startup against a fixed simulated date so every run gives identical results.
Run from the project root:  python mcp_server/contract_shim.py
"""

import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIMULATED_TODAY = date(2025, 4, 1)
INT_FIELDS = ("annual_value_inr", "notice_days", "seats_purchased",
              "seats_active", "proposed_uplift_pct")
PATCHABLE = ("status", "owner", "proposed_uplift_pct")

app = Flask(__name__)
CONTRACTS: dict[str, dict] = {}  # keyed by upper-case contract_id


def notice_state(deadline: date, renewal: date, today: date = SIMULATED_TODAY) -> str:
    if renewal < today:
        return "EXPIRED"
    if deadline <= today:          # deadline passed, renewal not yet
        return "INSIDE_WINDOW"
    if (deadline - today).days <= 30:
        return "APPROACHING"
    return "OPEN"


def approval_band(value: int) -> str:
    if value < 1_000_000:
        return "A"
    return "B" if value <= 5_000_000 else "C"


def enrich(row: dict) -> dict:
    for f in INT_FIELDS:
        row[f] = int(row[f])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"
    renewal = date.fromisoformat(row["renewal_date"])
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["notice_state"] = notice_state(deadline, renewal)
    # AMC / support contracts have no seats, so utilisation is undefined (null)
    row["utilisation_pct"] = (round(100 * row["seats_active"] / row["seats_purchased"])
                              if row["seats_purchased"] > 0 else None)
    row["approval_band"] = approval_band(row["annual_value_inr"])
    return row


def load_contracts() -> None:
    CONTRACTS.clear()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = enrich(row)
            CONTRACTS[c["contract_id"].upper()] = c


def not_found(contract_id: str):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "mock-contract-api",
                    "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    results = list(CONTRACTS.values())
    filters = {"category": "category", "band": "approval_band",
               "notice_state": "notice_state"}
    for param, field in filters.items():
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.strip().lower()]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    try:
        days = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400
    results = sorted((c for c in CONTRACTS.values() if 0 <= c["days_to_renewal"] <= days),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": days, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    c = CONTRACTS.get(contract_id.upper())
    return jsonify(c) if c else not_found(contract_id)


@app.get("/api/categories")
def categories():
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS.values():
        grouped.setdefault(c["category"], []).append(
            {k: c[k] for k in ("contract_id", "vendor", "annual_value_inr", "utilisation_pct")})
    summary = [{"category": cat, "vendor_count": len(items),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
                "vendors": items}
               for cat, items in sorted(grouped.items())]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only: restarting the server resets every change."""
    c = CONTRACTS.get(contract_id.upper())
    if not c:
        return not_found(contract_id)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    updates = {k: v for k, v in payload.items() if k in PATCHABLE}
    if not updates:
        return jsonify({"error": f"Nothing to update; allowed fields: {list(PATCHABLE)}"}), 400
    if "proposed_uplift_pct" in updates:
        try:
            updates["proposed_uplift_pct"] = float(updates["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return jsonify({"error": "proposed_uplift_pct must be a number"}), 400
    c.update(updates)
    return jsonify({"updated": True, "fields": sorted(updates), "contract": c})


load_contracts()  # once, at import/startup

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Mock Contract API - {len(CONTRACTS)} contracts from {CSV_PATH.name}")
    print(f"  Simulated today: {SIMULATED_TODAY}")
    print("  http://localhost:5001/health")
    print("=" * 60)
    app.run(port=5001, debug=False)