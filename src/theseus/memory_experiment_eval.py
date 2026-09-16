"""A small experiment-specific memory benchmark, offline unless explicitly configured.

Offline mode uses labeled reference extractions and lexical retrieval. It tests
storage and retrieval mechanics, not LLM quality. Live mode can evaluate a chosen
extractor, embedder, and optionally answers against the same fixed scenarios.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from theseus.json_utils import parse_json_response
from theseus.layer_store import lexical_score, load_lines
from theseus.memory_module import MemoryModule, Episode
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.stimulus_log import StimulusLog


# Source events and reference extractions are intentionally separate from queries.
# Every event carries who reported it; no actual clients or payments are involved.
SCENARIOS = (
    ("human", "Atlas prototype deadline is Friday.", "Atlas", "prototype deadline", "Friday"),
    ("Alpha", "I own Atlas prototype delivery.", "Atlas", "delivery owner", "Alpha"),
    ("human", "Atlas moved the prototype deadline from Friday to Monday.", "Atlas", "prototype deadline", "Monday"),
    ("Beta", "I accepted the Atlas delivery handoff from Alpha.", "Atlas", "delivery owner", "Beta"),
    ("Alpha", "Our acquisition approach is cold-email outreach.", "Project", "acquisition approach", "cold-email outreach"),
    ("human", "Stop cold-email outreach. Use paid pilot projects for acquisition.", "Project", "acquisition approach", "paid pilot projects"),
    ("human", "Atlas agreed to a prototype quote of 180 CAD.", "Atlas", "prototype quote", "180 CAD"),
    ("tool", "Atlas payment attempt failed. No funds were received.", "Atlas", "payment status", "unconfirmed"),
    ("Beta", "I plan to back up the project. I have not run the backup yet.", "Project", "backup status", "planned"),
    ("human", "The Atlas agreement covers a prototype, not a production launch.", "Atlas", "delivery scope", "prototype"),
    ("Alpha", "Boreal's deadline is Thursday. Its delivery owner is Alpha.", "Boreal", "deadline", "Thursday"),
    ("tool", "Saved Atlas prototype at /work/atlas/prototype.html.", "Atlas", "prototype artifact", "/work/atlas/prototype.html"),
    ("human", "Boreal requires a Spanish demo.", "Boreal", "demo language", "Spanish"),
    ("tool", "Boreal invoice is a local draft. It has not been sent.", "Boreal", "invoice status", "draft"),
)

QUERIES = (
    ("What is Atlas's current prototype deadline?", "Monday", ("Friday",)),
    ("Who currently owns Atlas delivery?", "Beta", ("Alpha",)),
    ("What is the project's current acquisition approach?", "paid pilot projects", ("cold-email outreach",)),
    ("What prototype quote did Atlas agree to?", "180 CAD", ()),
    ("What is Atlas's payment status?", "unconfirmed", ("paid", "received payment")),
    ("What is the project backup status?", "planned", ("completed", "successful")),
    ("What is Atlas's agreed delivery scope?", "prototype", ("production launch",)),
    ("When is the Boreal deadline?", "Thursday", ("Monday",)),
    ("Where is the saved Atlas prototype artifact?", "/work/atlas/prototype.html", ()),
    ("What language does Boreal require for its demo?", "Spanish", ("English",)),
    ("What is Boreal's invoice status?", "draft", ("sent", "paid")),
    ("What is Boreal's bank account number?", None, ()),
)


class ReferenceExtractor:
    response = ""

    reconciliation_response = '{"decisions": []}'

    def chat(self, prompt, **kwargs):
        if "<existing_knowledge>" in prompt:
            return self.reconciliation_response
        return self.response


def _score(text, expected, forbidden):
    lower = text.casefold()
    return {
        "expected_text_present": expected.casefold() in lower if expected else lower.strip() == "unknown",
        "forbidden_text_present": any(term.casefold() in lower for term in forbidden),
    }


def run_evaluation(workdir: Path, *, extractor=None, embedder=None, answerer=None) -> dict:
    workdir = Path(workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError("use an empty workdir so trials cannot reuse earlier memories")
    workdir.mkdir(parents=True, exist_ok=True)
    reference = extractor is None
    extractor = extractor or ReferenceExtractor()
    log = StimulusLog(workdir / "stimulus.jsonl")
    memory = MemoryModule(workdir / "memory", log, model_providers=[extractor],
                          embedding_providers=[embedder] if embedder else [])
    raw_records = []
    started = time.monotonic()
    for index, (actor, message, subject, predicate, value) in enumerate(SCENARIOS):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
        event = log.append(actor, "chat_message", {"message": message}, ts=timestamp)
        if reference:
            # The offline control supplies labeled reconciliation as well as
            # extraction; it must not depend on production failure fallbacks.
            current = memory.knowledge.current(subject, predicate)
            extractor.reconciliation_response = json.dumps({"decisions": [{
                "candidate_index": 0, "decision": "replace" if current else "new",
                "target_id": current[0].id if current else None,
            }]})
            extractor.response = json.dumps({"summary": f"{actor} reported: {message}", "assertions": [{
                "kind": "fact", "subject": subject, "predicate": predicate, "value": value,
                "statement": f"{subject} {predicate}: {value}",
                "support_event_ids": [event.id],
                "attribution": "direct_observation" if actor == "tool" else "partner_report",
                **({"reported_by": actor} if actor != "tool" else {}),
                "action_status": {7: "failure", 8: "intention", 11: "confirmed_outcome"}.get(index, "not_applicable"),
            }]})
        memory.consolidate(Episode(f"scenario-{index}", event.id, event.id))
        raw_records.append({"id": event.id, "text": message, "ordinal": index})
    # Remove all evidence from the live tail. Stored memories must carry the answers.
    for index in range(30):
        log.append("agent", "decision", {"text": f"Routine housekeeping {index}"})
    memory = MemoryModule(workdir / "memory", log, model_providers=[extractor],
                          embedding_providers=[embedder] if embedder else [])
    rows = []
    answer_calls = 0
    answer_usage = []
    for query, expected, forbidden in QUERIES:
        began = time.monotonic()
        recalled = memory.recall(query, budget_tokens=2000)
        layered = [{"id": entry.provenance.record_id, "text": entry.text}
                   for entry in recalled.entries[:3]]
        flat = sorted(raw_records, key=lambda row: (lexical_score(query, row["text"]), row["ordinal"]), reverse=True)
        flat = [{"id": row["id"], "text": row["text"]} for row in flat
                if lexical_score(query, row["text"]) > 0][:3]
        row = {"query": query, "expected": expected, "forbidden": forbidden,
               "recall_seconds": time.monotonic() - began, "systems": {}}
        for name, candidates in (("layered", layered), ("raw_lexical", flat)):
            top = candidates[0]["text"] if candidates else "unknown"
            outcome = {"candidates": candidates, "top1": _score(top, expected, forbidden),
                       "top3_answer_hit": any(expected.casefold() in c["text"].casefold() for c in candidates) if expected else not candidates}
            if answerer is not None:
                answer_calls += 1
                raw = answerer.chat(
                    "Answer from these memory records. Prefer current facts over historical reports. "
                    "Plans and attempts do not establish completed actions. If unsupported, answer unknown. "
                    "Return JSON with answer and evidence_ids.\n"
                    + json.dumps({"query": query, "records": candidates}), max_tokens=512,
                )
                usage = getattr(answerer, "last_chat_usage", None)
                if isinstance(usage, dict):
                    answer_usage.append(dict(usage))
                try:
                    answer = parse_json_response(raw)
                    if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                        raise ValueError("invalid answer object")
                    ids = answer.get("evidence_ids", [])
                    outcome["answer"] = {"text": answer["answer"], **_score(answer["answer"], expected, forbidden),
                                         "valid_citations": isinstance(ids, list) and all(rid in {c["id"] for c in candidates} for rid in ids)}
                except (ValueError, TypeError):
                    outcome["answer"] = {"error": "invalid JSON answer"}
            row["systems"][name] = outcome
        rows.append(row)
    traces = [json.loads(line) for line in load_lines(workdir / "memory" / "traces" / "consolidation.jsonl")]
    summary = {}
    for name in ("layered", "raw_lexical"):
        summary[name] = {
            "top1_expected": sum(row["systems"][name]["top1"]["expected_text_present"] for row in rows),
            "top3_expected": sum(row["systems"][name]["top3_answer_hit"] for row in rows),
            "top1_forbidden": sum(row["systems"][name]["top1"]["forbidden_text_present"] for row in rows),
        }
    report = {
        "mode": "reference-extraction-offline" if reference else "live-extraction",
        "extractor": getattr(extractor, "model", type(extractor).__name__),
        "embedding": getattr(embedder, "model", None), "queries": len(rows), "episodes": len(SCENARIOS),
        "summary": summary, "wall_seconds": time.monotonic() - started,
        "costs": {"extraction_chat_calls": sum(t["chat_calls"] for t in traces),
                  "answer_chat_calls": answer_calls,
                  "embedding_calls_at_consolidation": sum(t["embedding_calls"] for t in traces),
                  "embedding_calls_at_recall": memory._embedding_calls,
                  "tokens_in_estimated": sum(t["tokens_in"] for t in traces),
                  "tokens_out_estimated": sum(t["tokens_out"] for t in traces),
                  "reported_answer_usage": answer_usage,
                  "reported_chat_usage": [u for t in traces for u in t["reported_chat_usage"]]},
        "limitations": "Exact text checks measure retrieval diagnostics, not semantic answer correctness. Offline extractions are reference labels. Raw lexical control pays no extraction cost. Review live answers and citations manually; reported usage may omit failed requests and provider retries.",
        "results": rows,
    }
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDER_REGISTRY))
    parser.add_argument("--model")
    parser.add_argument("--embedding-provider", choices=sorted(PROVIDER_REGISTRY))
    parser.add_argument("--embedding-model")
    parser.add_argument("--answers", action="store_true", help="Also make live answer calls for both retrieval systems")
    args = parser.parse_args()
    if bool(args.provider) != bool(args.model) or bool(args.embedding_provider) != bool(args.embedding_model):
        parser.error("provide both a provider and its model")
    if (args.answers or args.embedding_provider) and not args.provider:
        parser.error("live embeddings/answers require a live extraction provider")
    extractor = PROVIDER_REGISTRY[args.provider](model=args.model) if args.provider else None
    embedder = PROVIDER_REGISTRY[args.embedding_provider](model=args.embedding_model) if args.embedding_provider else None
    report = run_evaluation(args.workdir, extractor=extractor, embedder=embedder,
                            answerer=extractor if args.answers else None)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
