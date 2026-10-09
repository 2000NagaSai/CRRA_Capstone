"""
CRRA Lab C4 - Portfolio Orchestrator

LangGraph StateGraph for contract renewal review:

    analysis -> policy_check --(hitl_required)--> hitl -> report
                             +--(otherwise)------------> report

An agent may recommend, but it may not commit Zensar to anything on its own:
policy_check (pure Python) decides when a named human must approve.

Prerequisites (from the project root):
    python data/kb_setup.py                 # Lab C1 - policy KB in data/chroma_db
    python mcp_server/contract_shim.py      # Lab C2 - keep running on port 5001
Run:
    python orchestrator/supervisor.py
"""

import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# `python orchestrator/supervisor.py` only puts orchestrator/ on the import path.
# Add the project root so `guardrails` resolves - before any project import.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic                                   # noqa: E402
import chromadb                                    # noqa: E402
import requests                                    # noqa: E402
from dotenv import load_dotenv                     # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger   # noqa: E402

load_dotenv(ROOT / ".env", override=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
CHROMA_DIR = ROOT / "data" / "chroma_db"
MAX_ROUNDS = 5

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
audit = AuditLogger()


def extract_text(response) -> str:
    """First block with text - a thinking block may come first."""
    for block in response.content:
        if hasattr(block, "text") and block.text:
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class ContractState(TypedDict):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str
    final_status: str


def new_state(contract_id: str) -> ContractState:
    return {"contract_id": contract_id, "contract": {}, "recommendation": "",
            "confidence": "", "rationale": "", "policy_citation": "",
            "estimated_annual_impact_inr": 0, "hitl_required": False, "hitl_reason": "",
            "hitl_approved": False, "approver": "", "final_status": ""}


# ══════════════════════════════════════════════════════════════
# POLICY KB (Lab C1)
# ══════════════════════════════════════════════════════════════
def _chunk(text: str, filename: str) -> list[tuple[str, str, dict]]:
    """One chunk per '## ' section; '# ' title skipped (same rules as Lab C1)."""
    out, heading, body = [], "Overview", []

    def flush():
        content = "\n".join(body).strip()
        if content:
            i = len(out)
            out.append((f"{Path(filename).stem}::{i:02d}", f"{heading}\n\n{content}",
                        {"source": filename, "heading": heading, "chunk_index": i}))

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            flush()
            heading, body = s[3:].strip(), []
        elif not s.startswith("# "):
            body.append(line)
    flush()
    return out


def load_kb():
    """Open the persisted collection from Lab C1; rebuild it if missing."""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        col = db.get_collection(KB_COLLECTION)
        if col.count() > 0:
            return col
        db.delete_collection(KB_COLLECTION)
    except Exception:
        pass
    col = db.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    chunks = [c for md in sorted((ROOT / "data" / "kb").glob("*.md"))
              for c in _chunk(md.read_text(encoding="utf-8"), md.name)]
    if not chunks:
        raise SystemExit("No policy files in data/kb/ - cannot build the KB.")
    ids, docs, metas = zip(*chunks)
    col.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
    print(f"  (policy KB was missing - rebuilt with {len(ids)} chunks)")
    return col


KB = None


def search_policy(query: str) -> list[dict]:
    """Best section per source file, top 2 files."""
    res = KB.query(query_texts=[query], n_results=min(8, KB.count()))
    best: dict[str, tuple] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta["source"] not in best or dist < best[meta["source"]][0]:
            best[meta["source"]] = (dist, meta["heading"], doc)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return [{"source": s, "section": h, "confidence": round(1 - d, 2), "text": t}
            for s, (d, h, t) in ranked]


# ══════════════════════════════════════════════════════════════
# NODE 1 - ANALYSIS (model + tools)
# ══════════════════════════════════════════════════════════════
ANALYSIS_TOOLS = [
    {"name": "search_policy",
     "description": "Search the procurement policy KB. Returns the best section from each "
                    "of the top 2 policy files. Call at most twice.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
    {"name": "submit_recommendation",
     "description": "Record the final recommendation. Call exactly once, last.",
     "input_schema": {
         "type": "object",
         "properties": {
             "recommendation": {"type": "string",
                                "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]},
             "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
             "rationale": {"type": "string",
                           "description": "Two or three sentences with the numbers that decided it."},
             "policy_citation": {"type": "string",
                                 "description": "e.g. 'renegotiation_levers.md §Price uplift benchmarks'"},
             "estimated_annual_impact_inr": {"type": "integer",
                                             "description": "Negative means a saving. 0 if unchanged."},
         },
         "required": ["recommendation", "confidence", "rationale", "policy_citation",
                      "estimated_annual_impact_inr"]}},
]

ANALYSIS_SYSTEM = """You are the Renewal Analysis Agent for Zensar BizOps.
Recommend exactly one of RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE for the contract given.
Search the policy KB (at most twice) for the governing rule, then call
submit_recommendation exactly once. You have at most 5 turns.

Guidance:
- Proposed uplift above 15% is never accepted at first offer -> RENEGOTIATE.
- Utilisation under 40% with an overlapping vendor in the same category -> CONSOLIDATE.
- Healthy utilisation and modest uplift -> RENEW.
- TERMINATE only if the capability is no longer needed by anyone. An UNASSIGNED owner
  is not proof of that - recommend conservatively and say why.
- utilisation_pct of null means the contract has no seats; it is not zero.

Confidence: HIGH = data and policy clearly agree; MEDIUM = sound but rests on a named
assumption; LOW = genuinely unclear. LOW confidence is a valid, useful answer - give it
honestly rather than inventing certainty or searching again for a cleaner picture.
Always cite the policy file and section."""

FACT_FIELDS = ("contract_id", "vendor", "category", "business_unit", "owner",
               "annual_value_inr", "approval_band", "renewal_date", "notice_deadline",
               "notice_state", "auto_renew", "seats_purchased", "seats_active",
               "utilisation_pct", "proposed_uplift_pct", "status")


def _fail(state, status: str, msg: str) -> dict:
    print(f"  x {msg}")
    audit.log("AnalysisAgent", "analysis_error", state["contract_id"], msg, "N/A")
    return {"final_status": status, "rationale": msg, "hitl_required": False}


def _low_confidence(cid: str, contract: dict, why: str) -> dict:
    print(f"  ! {why} - recording LOW confidence for human review.")
    audit.log("AnalysisAgent", "analysis_incomplete", cid, why, "N/A")
    return {"contract": contract, "recommendation": "RENEGOTIATE", "confidence": "LOW",
            "rationale": f"{why}. Needs manual review.", "policy_citation": "n/a",
            "estimated_annual_impact_inr": 0}


def analysis_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    print(f"\n{'═' * 66}\nCONTRACT: {cid}\n{'═' * 66}\n\n▶ ANALYSIS AGENT")

    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{cid}", timeout=10)
    except requests.exceptions.RequestException:
        return _fail(state, "ERROR_API_UNREACHABLE",
                     "Contract API unreachable - is mcp_server/contract_shim.py running on port 5001?")
    if r.status_code == 404:
        return _fail(state, "ERROR_NOT_FOUND", f"Contract {cid} does not exist")
    contract = r.json()

    util = contract.get("utilisation_pct")
    print(f"  {contract['vendor']} · {contract['category']} · band {contract['approval_band']} · "
          f"owner {contract['owner']}\n"
          f"  INR {contract['annual_value_inr']:,}/yr · util {'n/a' if util is None else f'{util}%'} · "
          f"uplift {contract['proposed_uplift_pct']}% · {contract['notice_state']}")
    audit.log("AnalysisAgent", "contract_fetched", cid,
              f"{contract['vendor']}, band {contract['approval_band']}, {contract['notice_state']}")

    facts = json.dumps({k: contract.get(k) for k in FACT_FIELDS}, indent=2)
    messages = [{"role": "user", "content": f"Analyse this contract:\n{facts}"}]

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                output_config={"effort": "medium"},   # temperature not supported
                system=ANALYSIS_SYSTEM,
                tools=ANALYSIS_TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            return _fail(state, "ERROR_MODEL", f"Anthropic API error: {e}")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            text = extract_text(response)
            return _low_confidence(cid, contract,
                                   f"Agent stopped without submitting ({text[:120] or response.stop_reason})")

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            if block.name == "submit_recommendation":
                a = block.input
                print(f"  → {a['recommendation']} (confidence {a['confidence']}) · {a['policy_citation']}")
                audit.log("AnalysisAgent", "recommendation", cid,
                          f"{a['recommendation']} / {a['confidence']} - {a['policy_citation']}",
                          "N/A", rounds_used=round_no)
                return {"contract": contract,
                        "recommendation": a["recommendation"],
                        "confidence": a["confidence"],
                        "rationale": a.get("rationale", ""),
                        "policy_citation": a.get("policy_citation", ""),
                        "estimated_annual_impact_inr": int(a.get("estimated_annual_impact_inr") or 0)}
            if block.name == "search_policy":
                hits = search_policy(block.input.get("query", ""))
                tops = ", ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})" for h in hits)
                print(f'  → policy "{block.input.get("query", "")[:40]}" -> {tops}')
                result = {"results": hits}
            else:
                result = {"error": f"unknown tool {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})

        if round_no == MAX_ROUNDS - 1:
            results.append({"type": "text", "text": "Final turn: call submit_recommendation "
                                                    "now. LOW confidence is acceptable."})
        messages.append({"role": "user", "content": results})

    return _low_confidence(cid, contract, f"No recommendation after {MAX_ROUNDS} rounds")


# ══════════════════════════════════════════════════════════════
# NODE 2 - POLICY CHECK (pure Python, no model call)
# ══════════════════════════════════════════════════════════════
def policy_check_node(state: ContractState) -> dict:
    print("\n▶ POLICY CHECK")
    contract = state.get("contract") or {}
    if state.get("final_status", "").startswith("ERROR") or not contract:
        audit.log("PolicyCheck", "skipped", state["contract_id"],
                  f"no contract data ({state.get('final_status') or 'unknown'})")
        return {"hitl_required": False, "hitl_reason": "not evaluated"}

    reasons: list[str] = []      # accumulate every trigger - never stop at the first
    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"approval band {band} requires a named human approver")
    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append("inside the notice window - leverage already lost")
    if state.get("recommendation") == "TERMINATE":
        reasons.append("all terminations require written owner confirmation")
    if state.get("confidence") == "LOW":
        reasons.append("analysis confidence is LOW")
    if str(contract.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("no business owner on record (owner UNASSIGNED)")

    hitl_required = bool(reasons)
    reason_text = "; ".join(reasons) if reasons else "no policy trigger"
    print(f"  HITL required: {hitl_required}")
    for r in reasons:
        print(f"    · {r}")
    audit.log("PolicyCheck", "evaluate_triggers", state["contract_id"], reason_text,
              "PENDING" if hitl_required else "N/A", triggers=len(reasons))
    return {"hitl_required": hitl_required, "hitl_reason": reason_text}


# ══════════════════════════════════════════════════════════════
# NODE 3 - HUMAN APPROVAL GATE
# ══════════════════════════════════════════════════════════════
def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:            # no interactive terminal -> treat as no answer
        return ""


def hitl_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    impact = state.get("estimated_annual_impact_inr") or 0
    print("\n▶ HUMAN APPROVAL GATE")
    print(f"  Contract : {cid}  ({state['contract'].get('vendor')})")
    print(f"  Proposed : {state['recommendation']}  (confidence {state['confidence']})")
    print(f"  Impact   : INR {impact:,}/yr")
    print(f"  Policy   : {state['policy_citation']}")
    print("  Why a human must decide:")
    for r in state["hitl_reason"].split("; "):
        print(f"    · {r}")
    print(f"\n  Rationale: {state['rationale']}")
    audit.log("HITLGate", "approval_request", cid,
              f"{state['recommendation']} - {state['hitl_reason']}", "PENDING")

    answer = ""
    while answer not in ("y", "n"):
        answer = _ask("\n  Approve this recommendation? [y/n]: ").lower()[:1]
        if answer == "":
            answer = "n"        # no input available -> safe default is reject
            print("  (no answer - treated as rejected)")
    approved = answer == "y"

    approver = ""
    if approved:
        while not approver:
            approver = _ask("  Approver name: ") or ""
            if not approver:
                print("  A name is required to approve.")
    decider = approver or (_ask("  Your name (for the audit trail): ") or "unknown")

    audit.log("HITLGate", "approval_decision", cid,
              f"{state['recommendation']} {'approved' if approved else 'rejected'}",
              "APPROVED" if approved else "REJECTED", actor=decider)
    print(f"  → {'APPROVED by ' + approver if approved else 'REJECTED by ' + decider + ' - no action taken'}")
    return {"hitl_approved": approved, "approver": decider}


# ══════════════════════════════════════════════════════════════
# NODE 4 - REPORT
# ══════════════════════════════════════════════════════════════
def report_node(state: ContractState) -> dict:
    print("\n▶ REPORT")
    cid = state["contract_id"]
    if state.get("final_status", "").startswith("ERROR"):
        audit.log("Reporting", "final_status", cid, state["final_status"], "N/A")
        print(f"  FINAL STATUS: {state['final_status']}")
        return {}

    rec = state["recommendation"]
    if not state.get("hitl_required"):
        final, note, status = f"{rec}_AUTO", "Actioned without human approval (no policy trigger).", "N/A"
    elif state.get("hitl_approved"):
        final, note, status = (f"{rec}_APPROVED",
                               f"Approved by {state.get('approver')}. Cleared to action.", "APPROVED")
    else:
        final, note, status = ("ON_HOLD_REJECTED",
                               "Rejected at the approval gate. No commitment made to the vendor.",
                               "REJECTED")
    audit.log("Reporting", "final_status", cid, f"{final} - {note}", status,
              actor=state.get("approver") or "system")
    print(f"  FINAL STATUS: {final}\n  {note}")
    return {"final_status": final}


# ══════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════
def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)
    g.add_edge(START, "analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges("policy_check", route_after_policy_check,
                            {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def check_api_key() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key.")
    try:
        client.models.list(limit=1)
    except anthropic.AuthenticationError:
        raise SystemExit(f"Anthropic rejected the API key (starts '{key[:10]}', length {len(key)}).")
    except anthropic.APIConnectionError as e:
        raise SystemExit(f"Cannot reach the Anthropic API: {e}")


def main() -> None:
    global KB
    check_api_key()
    KB = load_kb()
    graph = build_graph()

    # CTR-1010 has no trigger and should finish with no prompt; the other two stop
    # for a human (approve CTR-1012, reject CTR-1006 in Step 4).
    portfolio = ["CTR-1010", "CTR-1012", "CTR-1006"]
    outcomes = [graph.invoke(new_state(cid)) for cid in portfolio]

    print(f"\n\n{'═' * 78}\nPORTFOLIO REVIEW COMPLETE\n{'═' * 78}")
    print(f"{'Contract':<10}{'Action':<14}{'Conf':<8}{'Gate':<7}{'Final status':<24}Approver")
    print("-" * 78)
    for o in outcomes:
        gate = "HUMAN" if o.get("hitl_required") else "auto"
        print(f"{o['contract_id']:<10}{o.get('recommendation') or '-':<14}"
              f"{o.get('confidence') or '-':<8}{gate:<7}{o.get('final_status') or '-':<24}"
              f"{o.get('approver') or '-'}")
    audit.summary()


if __name__ == "__main__":
    main()
