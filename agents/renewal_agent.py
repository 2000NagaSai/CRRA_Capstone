"""
CRRA Lab C3 - Renewal Analysis Agent

Given a contract ID, decides RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE with a
confidence level and a citation to the procurement policy section that supports it.

Prerequisites (separate terminals, from the project root):
    python data/kb_setup.py                 # Lab C1 - builds data/chroma_db
    python mcp_server/contract_shim.py      # Lab C2 - leave running on port 5001

Run from the project root:
    python agents/renewal_agent.py
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
try:  # box-drawing characters on a Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_DIR = ROOT / "data" / "kb"
CHROMA_DIR = ROOT / "data" / "chroma_db"   # same store Lab C1 writes to
KB_COLLECTION = "crra_policy"
MAX_ROUNDS = 5          # hard cap so a model that will not converge cannot loop forever
MAX_TOKENS = 4000       # room for adaptive thinking plus the tool call
VALID_RECS = {"RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"}

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


# ══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_text(response) -> str:
    """First block that has text. A thinking block may come first, so
    response.content[0].text is not safe."""
    for block in response.content:
        if getattr(block, "text", None):
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE (Lab C1)
# ══════════════════════════════════════════════════════════════
_kb = None


def _chunk(text: str, filename: str) -> list[tuple[str, str, dict]]:
    """Fallback chunker (same rules as Lab C1): one chunk per '## ' section."""
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


def get_kb():
    """Open the persisted 'crra_policy' collection; rebuild it if it is missing."""
    global _kb
    if _kb is not None:
        return _kb
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        _kb = db.get_collection(KB_COLLECTION)
        if _kb.count() > 0:
            return _kb
        db.delete_collection(KB_COLLECTION)
    except Exception:
        pass
    print("  KB collection not found - building it from data/kb/ ...")
    _kb = db.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    chunks = [c for md in sorted(KB_DIR.glob("*.md"))
              for c in _chunk(md.read_text(encoding="utf-8"), md.name)]
    if not chunks:
        raise SystemExit(f"No policy .md files found in {KB_DIR}")
    ids, docs, metas = zip(*chunks)
    _kb.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
    print(f"  Built KB with {len(ids)} chunks.")
    return _kb


# ══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════
TOOLS = [
    {
        "name": "get_contract",
        "description": ("Fetch one contract by ID, including derived fields: approval_band, "
                        "notice_state, notice_deadline, utilisation_pct (null for contracts "
                        "with no seats), days_to_renewal, proposed_uplift_pct, owner."),
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": ("Search the BizOps procurement policy knowledge base. Returns the best "
                        "matching section from each of the top 2 policy files, with a 0-1 "
                        "confidence. Use it to find the rule that governs your recommendation. "
                        "Call at most twice per contract."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {
                "type": "string",
                "description": "Plain-English policy question, e.g. 'who approves a 60 lakh contract'"}},
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": ("List every contract in a category with vendor, annual value and "
                        "utilisation. Use only when judging whether CONSOLIDATE is viable."),
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation for this contract. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string", "enum": sorted(VALID_RECS)},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {"type": "string", "description":
                              "Two or three sentences citing the specific numbers that drove the decision."},
                "policy_citation": {"type": "string", "description":
                                    "Policy file and section, e.g. 'auto_renewal_rules.md §Escalation trigger'"},
                "estimated_annual_impact_inr": {"type": "integer", "description":
                                                "Rough rupee impact per year. 0 for RENEW at existing "
                                                "terms. Negative means a saving."},
                "human_approval_required": {"type": "boolean", "description":
                                            "True if policy requires a named human to approve before acting."},
            },
            "required": ["contract_id", "recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr",
                         "human_approval_required"],
        },
    },
]


# ══════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS (never raise - errors go back to the model as data)
# ══════════════════════════════════════════════════════════════
def _api_get(path: str):
    try:
        r = requests.get(f"{CONTRACT_API}{path}", timeout=10)
    except requests.exceptions.RequestException:
        return None, {"error": ("Contract API unreachable at localhost:5001. Is "
                                "mcp_server/contract_shim.py running? Do not guess the data.")}
    if r.status_code == 404:
        return None, {"error": "not_found"}
    if not r.ok:
        return None, {"error": f"Contract API returned HTTP {r.status_code}"}
    return r.json(), None


def tool_get_contract(contract_id: str) -> dict:
    data, err = _api_get(f"/api/contracts/{contract_id}")
    if err and err["error"] == "not_found":
        return {"error": f"Contract {contract_id} does not exist. Do not invent its data."}
    return err or data


def tool_search_policy(query: str) -> dict:
    """Best section per source file, top 2 files."""
    try:
        kb = get_kb()
        res = kb.query(query_texts=[query], n_results=min(8, kb.count()),
                       include=["documents", "metadatas", "distances"])
    except Exception as e:
        return {"error": f"Policy KB query failed: {e}"}
    best: dict[str, tuple] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta["source"]
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta["heading"], doc)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return {"results": [{"source": src, "section": heading,
                         "confidence": round(1 - dist, 2), "text": doc}
                        for src, (dist, heading, doc) in ranked]}


def tool_find_category_overlap(category: str) -> dict:
    data, err = _api_get("/api/categories")
    if err:
        return err
    for entry in data.get("categories", []):
        if entry["category"].strip().lower() == category.strip().lower():
            return entry
    return {"category": category, "vendor_count": 0, "vendors": [],
            "note": "No contracts in this category."}


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps.

For the contract you are given, recommend exactly one of: RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE.

Method, in order:
1. Call get_contract to read the real numbers. Never assume or invent them. If the tool
   returns an error (contract missing, API unreachable), do not guess: submit with
   confidence LOW, human_approval_required true, and explain the error in the rationale.
2. Call search_policy to find the governing rule. At most twice.
3. Call find_category_overlap ONLY if you are considering CONSOLIDATE.
4. Call submit_recommendation exactly once to finish. You have at most 5 turns in total.

Decision guidance:
- RENEW: healthy utilisation and an uplift within normal range.
- RENEGOTIATE: still needed but the terms are wrong. Uplift above 15% is never accepted
  at first offer; above 8% should be challenged.
- CONSOLIDATE: utilisation below 40% AND a viable overlapping vendor in the same category.
  The capability is still needed, just delivered through another existing agreement.
- TERMINATE: only when the capability itself is no longer required by anyone, and the
  business owner has confirmed it. Low utilisation alone is not enough.
- An UNASSIGNED or missing owner is NOT evidence the capability is unneeded. Do not
  recommend a confident TERMINATE; escalate to a human with LOW or MEDIUM confidence.
- utilisation_pct may be null (AMC or support contracts with no seats). Do not treat
  null as zero.
- Being INSIDE_WINDOW weakens leverage; say so in the rationale rather than hiding it.

Confidence:
- HIGH: the numbers and the policy point the same way with no ambiguity.
- MEDIUM: sound, but rests on an assumption you name in the rationale.
- LOW: genuinely unclear. LOW confidence is a valid, useful answer. Say so honestly
  rather than inventing certainty, and do not keep calling tools hoping for a cleaner picture.

human_approval_required must be true for approval Band B or C, anything INSIDE_WINDOW,
every TERMINATE, and any LOW-confidence result.

policy_citation must name the file and section returned by search_policy, e.g.
'termination_procedure.md §When termination is the right recommendation'."""


# ══════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════
def _fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v}%"


def _run_tool(name: str, args: dict) -> dict:
    if name == "get_contract":
        result = tool_get_contract(args.get("contract_id", ""))
        if "error" in result:
            print(f"  x {result['error']}")
        else:
            print(f"  -> contract: band {result['approval_band']}  {result['notice_state']}  "
                  f"util {_fmt_pct(result['utilisation_pct'])}  "
                  f"uplift {result['proposed_uplift_pct']}%  INR {result['annual_value_inr']:,}  "
                  f"owner {result.get('owner')}")
        return result
    if name == "search_policy":
        result = tool_search_policy(args.get("query", ""))
        tops = ", ".join(f"{r['source']} §{r['section']} ({r['confidence']:.0%})"
                         for r in result.get("results", [])) or result.get("error", "")
        print(f'  -> policy "{args.get("query", "")[:50]}"\n       {tops}')
        return result
    if name == "find_category_overlap":
        result = tool_find_category_overlap(args.get("category", ""))
        names = ", ".join(f"{v['vendor']} ({_fmt_pct(v['utilisation_pct'])})"
                          for v in result.get("vendors", []))
        print(f"  -> overlap in {args.get('category')}: {names or result.get('error', 'none')}")
        return result
    return {"error": f"Unknown tool {name}"}


def _print_recommendation(rec: dict) -> None:
    print("\n  ┌─ RECOMMENDATION ─────────────────────────────────")
    print(f"  │ {rec['recommendation']}   confidence {rec['confidence']}")
    print(f"  │ Policy: {rec.get('policy_citation', '-')}")
    print(f"  │ Impact: INR {rec.get('estimated_annual_impact_inr') or 0:,}/yr")
    print(f"  │ Human approval required: {rec['human_approval_required']}")
    print("  └──────────────────────────────────────────────────")
    print(f"  {rec.get('rationale', '')}")


def analyse_contract(contract_id: str) -> dict | None:
    """Run the agent on one contract. Returns the submitted recommendation, or None
    if the agent stopped without submitting (round cap, refusal, or plain-text reply)."""
    print(f"\n{'═' * 62}\nANALYSING: {contract_id}\n{'═' * 62}")
    messages = [{"role": "user",
                 "content": f"Analyse contract {contract_id} and recommend an action."}]

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            response = get_client().messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "medium"},   # no temperature on this model
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"  x Anthropic API error: {e}")
            return None

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            print(f"  Agent stopped without a recommendation ({response.stop_reason}).")
            if text := extract_text(response):
                print(f"  Agent said: {text[:300]}")
            return None

        # Keep the whole content (including any thinking blocks) in the history
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            if block.name == "submit_recommendation":
                rec = dict(block.input)
                if rec.get("recommendation") not in VALID_RECS:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"recommendation must be one of {sorted(VALID_RECS)}"})
                    continue
                rec["contract_id"] = rec.get("contract_id") or contract_id
                rec["rounds_used"] = round_no
                _print_recommendation(rec)
                return rec
            result = _run_tool(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                            **({"is_error": True} if "error" in result else {})})

        if round_no == MAX_ROUNDS - 1:  # nudge before the last turn
            results.append({"type": "text", "text": (
                "Final turn: call submit_recommendation now with what you have. "
                "LOW confidence is an acceptable answer.")})
        messages.append({"role": "user", "content": results})

    print(f"  ! Stopped after {MAX_ROUNDS} rounds with no recommendation - "
          "treat as LOW confidence and escalate to a human.")
    return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key.")
    if _api_get("/health")[1]:
        print("WARNING: contract API on localhost:5001 is not responding. "
              "Start mcp_server/contract_shim.py in another terminal.\n")

    # A deliberate spread: healthy renewal, high uplift, low utilisation with an
    # overlap, an orphaned contract, and a band C contract inside its notice window.
    test_contracts = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]

    rows = []
    for cid in test_contracts:
        rec = analyse_contract(cid)
        rows.append(rec or {"contract_id": cid, "recommendation": "NO DECISION",
                            "confidence": "LOW", "human_approval_required": True,
                            "estimated_annual_impact_inr": None, "policy_citation": "-"})

    w = 112
    print(f"\n\n{'═' * w}\nPORTFOLIO SUMMARY\n{'═' * w}")
    print(f"{'Contract':<10}{'Action':<14}{'Conf':<8}{'Human?':<8}{'Impact (INR/yr)':>16}   Policy citation")
    print("-" * w)
    for r in rows:
        impact = r.get("estimated_annual_impact_inr")
        impact_s = "-" if impact is None else f"{impact:,}"
        cite = (r.get("policy_citation") or "-")[:54]
        print(f"{r['contract_id']:<10}{r['recommendation']:<14}{r['confidence']:<8}"
              f"{'YES' if r['human_approval_required'] else 'no':<8}{impact_s:>16}   {cite}")
    print("-" * w)
    decided = sum(r["recommendation"] != "NO DECISION" for r in rows)
    needs_human = sum(bool(r["human_approval_required"]) for r in rows)
    print(f"{decided}/{len(rows)} decided · {needs_human} require human approval before action")


if __name__ == "__main__":
    main()