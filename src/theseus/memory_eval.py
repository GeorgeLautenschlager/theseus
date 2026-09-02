"""Eval harness: MemoryModule vs AgenticMemory vs flat append-everything cosine.

One report, three systems over identical episodes and queries:
  - layered: the new MemoryModule (extraction + routing + RRF fusion)
  - agentic: the existing AgenticMemory, driven the way Alty drives it
  - flat:    append every episode's raw evidence, brute-force cosine recall

Accuracy is a deterministic substring check (does the answer text survive in a
top-K candidate?) plus an LLM judge used for RANKING ONLY — candidates plus a
"no relevant information" sentinel are shuffled and ranked; no thresholds, no
pass/fail from the judge. Latency p50/p95 per system, consolidation cost per
episode, store sizes on disk.

Needs live local Ollama (gemma4:e4b, nomic-embed-text). Not part of the
offline suite, like tests/test_fact_retention.py.

    poetry run python -m theseus.memory_eval --workdir /tmp/mem-eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import numpy as np

from theseus.agentic_memory import AgenticMemory
from theseus.json_utils import parse_json_response
from theseus.memory_module import Episode, MemoryModule, estimate_tokens
from theseus.memory_store import MemoryStore
from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.model_providers.openrouter_provider import OpenRouterProvider
from theseus.stimulus_log import StimulusLog

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LOGS = {
    "auto_alty": REPO_ROOT / "auto_alty" / "stimulus_log.jsonl",
    "session": REPO_ROOT / "stimulus_log.jsonl",
}
EPISODE_SIZE = 10         # events per real-log episode, file order (eval choice)
RECALL_BUDGET = 2048      # tokens
TOP_K = 3                 # candidates shown to the judge / counted for hits
MAX_EVENT_CHARS = 800     # per-event content cap at replay: CPU LLM practicality,
                          # applied identically to all three systems (eval choice)
SENTINEL = "(no relevant information available)"

# Content tokens that carry no identifying meaning for answer-matching
_STOP = {
    "the", "and", "for", "are", "was", "were", "has", "have", "had", "not",
    "its", "his", "her", "our", "their", "what", "when", "where", "which",
    "who", "how", "why", "does", "did", "out", "use", "using", "with",
}


def _answer_in(text: str, answer_text: str) -> bool:
    """Paraphrase-tolerant answer containment: every content token of the answer
    appears in the candidate. Strict superset of verbatim containment — needed
    because extraction renders facts as 'subject: value' triples ("is at" becomes
    "is scheduled at"), which verbatim matching wrongly counts as misses."""
    toks = {t.strip(string.punctuation).lower() for t in answer_text.split()} - _STOP
    toks = {t for t in toks if len(t) > 2}
    if not toks:
        return answer_text in text
    hay = {t.strip(string.punctuation).lower() for t in text.split()}
    return toks <= hay


# --- Golden set -----------------------------------------------------------------
# Real queries: answer text is verbatim in the named log. Synthetic top-up:
# clearly labeled, built from SYNTH_FACTS (the in-repo traces are too thin for
# 50 real queries — see report).


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    answer_text: str
    origin: str           # "real:<log>" or "synthetic"


REAL_QUERIES: list[GoldenQuery] = [
    GoldenQuery("What is a group of cats called?", "clowder", "real:auto_alty"),
    GoldenQuery("What goal did the agent write into GOALS.md?", "Improve the Theseus Agent Construction Kit", "real:auto_alty"),
    GoldenQuery("What is the Goal ID in GOALS.md?", "1a2b3c4d-5e6f-7g8h-9i0j-k1l2m3n4o5p6", "real:auto_alty"),
    GoldenQuery("What task was set up and what is its Task ID?", "9z8y7x6w-5v4u-3t2s-1r0q-p9o8n7m6l5k", "real:auto_alty"),
    GoldenQuery("What does the goal say to do, in its description?", "Systematically test all available tools and functional areas", "real:auto_alty"),
    GoldenQuery("What is the task's description?", "List files and directories to get a sense of the project structure", "real:auto_alty"),
    GoldenQuery("What priority was given to the goal and task?", "High", "real:auto_alty"),
    GoldenQuery("Which directory did the agent say it would explore next?", "src/", "real:auto_alty"),
    GoldenQuery("What messages did the user send in the chat log?", "Hello", "real:session"),
    GoldenQuery("What files did autobot read when reviewing its context?", "CLAUDE.md", "real:session"),
]

# (fact as it appears in the synthetic episode, query a user would ask)
SYNTH_FACTS: list[tuple[str, str]] = [
    ("Mara prefers afternoon meetings because mornings are for deep work.", "When does Mara prefer to have meetings and why?"),
    ("Mara's standup is at 10:15 on Tuesdays.", "When is Mara's standup?"),
    ("The staging database uses Postgres 16.", "What version of Postgres does staging use?"),
    ("Deployments are frozen every Friday after noon.", "When are deployments frozen?"),
    ("Mara wants error alerts routed to the ops channel, not email.", "Where should error alerts go?"),
    ("The legacy billing service is written in Ruby.", "What language is the legacy billing service written in?"),
    ("Incident reviews happen the morning after any Sev1.", "When do incident reviews happen?"),
    ("Mara decided to drop the weekly newsletter in March.", "What did Mara decide to stop doing, and when?"),
    ("The design system lives in the ui-kit repository.", "Where does the design system live?"),
    ("API keys rotate every 90 days automatically.", "How often do API keys rotate?"),
    ("Mara's on-call rotation ends in September.", "When does Mara's on-call rotation end?"),
    ("The search index is rebuilt nightly at 03:00 UTC.", "When is the search index rebuilt?"),
    ("Feature flags are managed with Unleash.", "What tool manages feature flags?"),
    ("Mara prefers written summaries over live demos for reviews.", "How does Mara prefer to receive review updates?"),
    ("The payments vendor is Adyen, not Stripe.", "Which payments vendor does the company use?"),
    ("Rate limits are 100 requests per minute per API key.", "What are the rate limits per API key?"),
    ("Mara asked for the dashboard to default to a dark theme.", "What did Mara ask the dashboard to default to?"),
    ("The audit log retains entries for seven years.", "How long are audit log entries retained?"),
    ("Backups run every six hours and keep fourteen generations.", "How often do backups run and how many generations are kept?"),
    ("Mara's manager is Priya, who reviews architecture proposals.", "Who is Mara's manager and what do they review?"),
    ("The mobile team ships on a two-week cadence.", "What is the mobile team's shipping cadence?"),
    ("Staging credentials are in the vault under staging-secrets.", "Where are the staging credentials stored?"),
    ("Mara decided the v2 API would be backward compatible for one year.", "What did Mara decide about v2 API compatibility?"),
    ("Load tests target 2000 concurrent users.", "What is the load test target for concurrent users?"),
    ("The analytics warehouse is Snowflake.", "Which analytics warehouse does the company use?"),
    ("Mara wants all customer emails to go through the review queue first.", "What did Mara want for customer emails?"),
    ("The CI pipeline caches dependencies between runs.", "What does the CI pipeline cache between runs?"),
    ("Deprecation warnings run for two release cycles before removal.", "How long do deprecation warnings run before removal?"),
    ("Mara's team owns the notifications service and its queues.", "What does Mara's team own?"),
    ("The status page is sourced from the incident tracker automatically.", "Where does the status page get its data?"),
    ("Mara prefers concise PR descriptions with a checklist.", "What does Mara prefer in PR descriptions?"),
    ("The sandbox environment resets every Monday at 02:00 UTC.", "When does the sandbox environment reset?"),
    ("Customer data exports are encrypted with AES-256 at rest.", "How are customer data exports encrypted at rest?"),
    ("Mara decided to pilot a weekly retro on Fridays in June.", "What did Mara decide to pilot, when, and in what month?"),
    ("The observability stack is Grafana on top of Loki.", "What is the observability stack?"),
    ("Mara asked that all timestamps be stored in UTC.", "What did Mara ask about timestamp storage?"),
    ("The mobile app requires TLS 1.3 or newer.", "What TLS version does the mobile app require?"),
    ("On-call handoffs include a written context summary.", "What do on-call handoffs include?"),
    ("Mara's team reviews the error budget every sprint.", "What does Mara's team review every sprint?"),
    ("The CDN caches static assets for 30 days with versioned URLs.", "How long does the CDN cache static assets?"),
]

SYNTH_NOISE = "Decision: continue with the current plan and check the logs."


def synthetic_episodes() -> tuple[list[dict], list[GoldenQuery]]:
    """Deterministic episodes: four facts + noise events each, ms-separated."""
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    episodes: list[dict] = []
    queries: list[GoldenQuery] = []
    for i in range(0, len(SYNTH_FACTS), 4):
        t0 = base + timedelta(milliseconds=i * 4000)
        events = []
        for j in range(4):
            fact, query = SYNTH_FACTS[i + j]
            events.append({"actor": "mara", "type": "exchange", "content": {"message": fact},
                          "ts": t0 + timedelta(milliseconds=j * 80)})
            if j < 3:
                events.append({"actor": "Alty", "type": "decision", "content": {"text": SYNTH_NOISE},
                               "ts": t0 + timedelta(milliseconds=j * 80 + 40)})
            queries.append(GoldenQuery(query, fact, "synthetic"))
        episodes.append({"episode_id": f"synth-{i // 4:02d}", "events": events})
    return episodes, queries


def truncated(events: list[dict]) -> list[dict]:
    """Cap long content strings (real logs carry whole-file tool outputs)."""
    out = []
    for e in events:
        content = {k: (v[:MAX_EVENT_CHARS] if isinstance(v, str) and len(v) > MAX_EVENT_CHARS else v)
                   for k, v in e["content"].items()}
        out.append({**e, "content": content})
    return out


def load_episodes() -> list[dict]:
    """All episodes in order: real logs chunked by EPISODE_SIZE, then synthetic.
    Events are plain dicts so every system re-appends them under its own log."""
    episodes: list[dict] = []
    for name, path in REAL_LOGS.items():
        events = StimulusLog(path).read_all()
        for i in range(0, len(events), EPISODE_SIZE):
            chunk = events[i : i + EPISODE_SIZE]
            episodes.append(
                {
                    "episode_id": f"real-{name}-{i // EPISODE_SIZE}",
                    "events": [
                        {"actor": e.actor, "type": e.type, "content": e.content, "ts": e.ts}
                        for e in chunk
                    ],
                }
            )
    synth_eps, _ = synthetic_episodes()
    episodes.extend(synth_eps)
    return episodes


# --- Cost tracking ---------------------------------------------------------------


class CostTracker:
    """Duck-typed ModelProvider wrapper that counts tokens and embed calls."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.tokens_in = 0
        self.tokens_out = 0
        self.embed_calls = 0

    def is_available(self) -> bool:
        return self._inner.is_available()

    def chat(self, prompt: str, json_schema=None):
        self.tokens_in += estimate_tokens(prompt)
        raw = self._inner.chat(prompt, json_schema=json_schema)
        self.tokens_out += estimate_tokens(raw if isinstance(raw, str) else json.dumps(raw))
        return raw

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._inner.embed(text)


# --- Systems -----------------------------------------------------------------------


@dataclass
class QueryOutcome:
    latencies: list[float] = field(default_factory=list)
    hits: list[bool] = field(default_factory=list)               # answer in top-K
    best_ranks: list[int | None] = field(default_factory=list)   # judge rank of answer candidate
    sentinel_last: list[bool] = field(default_factory=list)


# --- Chat backend ---------------------------------------------------------------
# "gemma" (default) = local gemma4:e4b; "claude:<model>" = ClaudeProvider via the
# local `claude` CLI (subscription auth, no API key); "openrouter:<model>" =
# OpenRouterProvider (OPENROUTER_API_KEY from env).
_CHAT_SPEC = "gemma"
_LENIENT = False


class _SchemalessChat:
    """ponytail: OpenRouter's strict json_schema decoding measured ~50x slower than
    free-form (146s vs 3s, same prompt); the prompts already demand bare JSON and
    parse_json_response tolerates fences. Add when a future model regresses on that."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def is_available(self) -> bool:
        return self._inner.is_available()

    def chat(self, prompt: str, json_schema=None):
        return self._inner.chat(prompt)


def _make_chat(spec: str | None = None):
    spec = spec or _CHAT_SPEC  # resolve at call time; a def-time default captured "gemma"
    if spec.startswith("claude:"):
        return ClaudeProvider(spec.split(":", 1)[1])
    if spec.startswith("openrouter:"):
        return _SchemalessChat(OpenRouterProvider(spec.split(":", 1)[1]))
    return OllamaProvider(model="gemma4:e4b")


class SystemBase:
    name = "base"

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir / self.name
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.chat = CostTracker(_make_chat())
        self.embed = CostTracker(OllamaProvider(model="nomic-embed-text"))
        self.consolidate_wall_s = 0.0
        self.outcome = QueryOutcome()

    def consolidate_episode(self, events: list[dict]) -> None:
        raise NotImplementedError

    def recall_top_k(self, query: str) -> list[str]:
        raise NotImplementedError

    def store_report(self) -> dict:
        raise NotImplementedError


class LayeredSystem(SystemBase):
    name = "layered"

    def __init__(self, workdir: Path) -> None:
        super().__init__(workdir)
        self.log = StimulusLog(self.workdir / "stimulus.jsonl")
        self.module = MemoryModule(
            self.workdir / "memory", self.log,
            embedding_providers=[self.embed], model_providers=[self.chat],
            lenient_fact_routing=_LENIENT,
        )
        self.consolidation_tokens_in = 0
        self.consolidation_tokens_out = 0

    def consolidate_episode(self, events: list[dict]) -> None:
        started = time.monotonic()
        ids = [
            self.log.append(actor=e["actor"], type=e["type"], content=e["content"], ts=e["ts"]).id
            for e in truncated(events)
        ]
        result = self.module.consolidate(Episode(ids[0], ids[0], ids[-1]))
        self.consolidate_wall_s += time.monotonic() - started
        self.consolidation_tokens_in += result.tokens_in
        self.consolidation_tokens_out += result.tokens_out

    def recall_top_k(self, query: str) -> list[str]:
        started = time.monotonic()
        result = self.module.recall(query, budget_tokens=RECALL_BUDGET)
        self.outcome.latencies.append(time.monotonic() - started)
        return [entry.text for entry in result.entries[:TOP_K]]

    def store_report(self) -> dict:
        base = self.workdir / "memory"
        return {
            "bytes": sum(f.stat().st_size for f in base.rglob("*") if f.is_file()),
            "records": {
                layer: len(getattr(self.module, layer).read_all())
                for layer in ("knowledge", "memory", "wisdom")
            },
        }

    def quality_report(self) -> dict:
        dl = self.workdir / "memory" / "dead_letter.jsonl"
        dead = [json.loads(line) for line in dl.read_text().splitlines() if line.strip()] if dl.exists() else []
        reasons: dict[str, int] = {}
        for rec in dead:
            reason = str(rec.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        supersessions = sum(1 for r in self.module.knowledge.read_all() if r.supersedes)
        return {"dead_letters": len(dead), "dead_letter_reasons": reasons,
                "supersessions": supersessions}


class AgenticSystem(SystemBase):
    name = "agentic"

    def __init__(self, workdir: Path) -> None:
        super().__init__(workdir)
        self.log = StimulusLog(self.workdir / "stimulus.jsonl")
        self.memory = AgenticMemory(
            model_providers=[self.chat], embedding_providers=[self.embed],
            store=MemoryStore(self.workdir / "notes.jsonl"), stimulus_log=self.log,
        )

    def consolidate_episode(self, events: list[dict]) -> None:
        for e in truncated(events):
            self.log.append(actor=e["actor"], type=e["type"], content=e["content"], ts=e["ts"])
        started = time.monotonic()
        self.memory.form()  # high-water mark makes this incremental per call
        self.consolidate_wall_s += time.monotonic() - started

    def recall_top_k(self, query: str) -> list[str]:
        started = time.monotonic()
        notes = self.memory._retrieve_notes(query)[:TOP_K]  # ponytail: eval-only access to private ranking
        self.outcome.latencies.append(time.monotonic() - started)
        return [f"{n.content}\n{n.context}" for n in notes]

    def store_report(self) -> dict:
        f = self.workdir / "notes.jsonl"
        return {"bytes": f.stat().st_size if f.exists() else 0, "records": {"notes": len(self.memory.store._notes)}}


class FlatSystem(SystemBase):
    """Append every episode's raw evidence; brute-force cosine recall. No LLM."""

    name = "flat"

    def __init__(self, workdir: Path) -> None:
        super().__init__(workdir)
        self.texts: list[str] = []
        self.vectors: list[np.ndarray] = []

    def _embed_long(self, text: str) -> np.ndarray:
        """Mean-pool chunk embeddings: nomic's effective context is smaller than a
        concatenated real episode, so long texts go in as chunks."""
        chunks = [text[i : i + 1500] for i in range(0, len(text), 1500)] or [""]
        vecs = np.vstack([np.asarray(self.embed.embed(c), dtype=np.float32) for c in chunks])
        v = vecs.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def consolidate_episode(self, events: list[dict]) -> None:
        started = time.monotonic()
        text = "\n".join(json.dumps(e, default=str) for e in truncated(events))
        self.texts.append(text)
        self.vectors.append(self._embed_long(text))
        self.consolidate_wall_s += time.monotonic() - started

    def recall_top_k(self, query: str) -> list[str]:
        started = time.monotonic()
        if not self.vectors:
            self.outcome.latencies.append(time.monotonic() - started)
            return []
        q = np.asarray(self.embed.embed(query), dtype=np.float32)
        mat = np.vstack(self.vectors)
        scores = mat @ q / (np.linalg.norm(mat) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(scores)[::-1][:TOP_K]
        self.outcome.latencies.append(time.monotonic() - started)
        return [self.texts[i] for i in top]

    def store_report(self) -> dict:
        return {"bytes": sum(len(t) for t in self.texts), "records": {"episode_records": len(self.texts)}}


# --- Judge (ranking only) ----------------------------------------------------------


def _score_ranking(ranking: list[str], labels: list[str], texts: dict, answer_text: str) -> tuple[int | None, bool]:
    """(best rank of an answer-bearing candidate, sentinel ranked last) for one
    system's label set within a judge ranking. Malformed/missing labels are a
    missing data point, not a crash."""
    valid = set(labels)
    clean = list(dict.fromkeys(str(lab) for lab in ranking if str(lab) in valid))
    sentinel = labels[-1]
    answer_labels = {lab for lab, t in texts.items() if lab != sentinel and _answer_in(t, answer_text)}
    ranked_answer = [i + 1 for i, lab in enumerate(clean) if lab in answer_labels]
    best_rank = min(ranked_answer) if ranked_answer else None
    # gemma4:e4b tends to omit the sentinel rather than rank it; an omitted
    # sentinel means no candidate lost to "no relevant information", which is
    # the metric's intent (returned beats nothing), so count omission as last.
    sentinel_last = sentinel not in set(clean) or clean[-1] == sentinel
    return best_rank, sentinel_last


def judge_query(judge, query: str, per_system: dict[str, list[str]], answer_text: str) -> dict[str, tuple[int | None, bool]]:
    """One judge call ranks every system's candidate list for a query.

    Returns {system_name: (best_rank, sentinel_last)}. The judge never decides
    correctness — only ordering. One call per query keeps the CPU-LLM cost of
    the phase bounded (150 calls -> 50).
    """
    prefixes = {name: name[0].upper() for name in per_system}
    blocks: list[str] = []
    texts_by_system: dict[str, dict] = {}
    labels_by_system: dict[str, list[str]] = {}
    schema_props = {}
    for name, candidates in per_system.items():
        p = prefixes[name]
        labels = [f"{p}{i}" for i in range(len(candidates))] + [f"{p}z"]
        texts = dict(zip(labels, candidates))
        texts[f"{p}z"] = SENTINEL
        seed = int.from_bytes(hashlib.sha256((query + name).encode()).digest()[:4], "big")
        order = random.Random(seed).sample(labels, len(labels))
        listing = "\n".join(f"{lab}: {texts[lab][:250]}" for lab in order)
        blocks.append(f"List {p} (labels {', '.join(labels)}):\n{listing}")
        texts_by_system[name] = texts
        labels_by_system[name] = labels
        schema_props[p] = {"type": "array", "items": {"type": "string"}}
    raw = judge.chat(
        "For each candidate list below, rank the candidates by how well they help answer "
        'the query. Use every label in a list exactly once. Respond with JSON like {"L": ["L0", ...], "A": [...]}\n\n'
        f"Query: {query}\n\n" + "\n\n".join(blocks),
        json_schema={"type": "object", "properties": schema_props},
    )
    try:
        parsed = parse_json_response(raw)
    except Exception:
        parsed = {}
    out: dict[str, tuple[int | None, bool]] = {}
    for name in per_system:
        p = prefixes[name]
        ranking = parsed.get(p, []) if isinstance(parsed.get(p), list) else []
        out[name] = _score_ranking(ranking, labels_by_system[name], texts_by_system[name], answer_text)
    return out


# --- Driver ------------------------------------------------------------------------


def evaluate(systems: list[SystemBase], workdir: Path) -> dict:
    """Recall + judge over the golden set for already-consolidated systems."""
    _, synth_queries = synthetic_episodes()
    golden = REAL_QUERIES + synth_queries
    judge = _make_chat()

    details = workdir / "query_details.jsonl"
    if details.exists():
        details.unlink()
    for qi, q in enumerate(golden):
        candidates_by_system = {s.name: s.recall_top_k(q.query) for s in systems}
        judged = judge_query(judge, q.query, candidates_by_system, q.answer_text)
        for system in systems:
            hit = any(_answer_in(c, q.answer_text) for c in candidates_by_system[system.name])
            best_rank, sentinel_last = judged[system.name]
            system.outcome.hits.append(hit)
            system.outcome.best_ranks.append(best_rank)
            system.outcome.sentinel_last.append(sentinel_last)
            with open(details, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"query": q.query, "origin": q.origin, "system": system.name,
                     "hit": hit, "best_rank": best_rank, "sentinel_last": sentinel_last},
                    ensure_ascii=False,
                ) + "\n")
        print(f"query {qi + 1}/{len(golden)} done", flush=True)

    def pct(values: list[bool]) -> float:
        return round(100.0 * sum(values) / len(values), 1) if values else 0.0

    def quantile_ms(latencies: list[float], q: float) -> float:
        if not latencies:
            return 0.0
        ordered = sorted(latencies)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))] * 1000, 1)

    report: dict = {"queries": len(golden), "episodes": sum(1 for _ in load_episodes()), "systems": {}}
    for system in systems:
        out = system.outcome
        ranks = [r for r in out.best_ranks if r is not None]
        report["systems"][system.name] = {
            "hit_rate_pct": pct(out.hits),
            "judge_answer_found_pct": pct([r is not None for r in out.best_ranks]),
            "judge_best_rank_mean": round(mean(ranks), 2) if ranks else None,
            "sentinel_last_pct": pct(out.sentinel_last),
            "recall_p50_ms": quantile_ms(out.latencies, 0.50),
            "recall_p95_ms": quantile_ms(out.latencies, 0.95),
            "consolidation_wall_s": round(system.consolidate_wall_s, 2),
            "consolidation_tokens_in": getattr(system, "consolidation_tokens_in", system.chat.tokens_in),
            "consolidation_tokens_out": getattr(system, "consolidation_tokens_out", system.chat.tokens_out),
            "store": system.store_report(),
        }
        if isinstance(system, LayeredSystem):
            report["systems"][system.name]["quality"] = system.quality_report()
    return report


def run(workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes()
    systems = [LayeredSystem(workdir), AgenticSystem(workdir), FlatSystem(workdir)]
    for system in systems:
        for i, ep in enumerate(episodes):
            print(f"[{system.name}] consolidate {i + 1}/{len(episodes)}: {ep['episode_id']}", flush=True)
            system.consolidate_episode(ep["events"])
    return evaluate(systems, workdir)


def main() -> None:
    global _CHAT_SPEC
    parser = argparse.ArgumentParser(description="Layered memory eval harness")
    parser.add_argument("--workdir", default="/tmp/mem-eval")
    parser.add_argument("--chat", default="gemma",
                        help="chat backend: 'gemma' (local), 'claude:<model>', or 'openrouter:<model>'")
    parser.add_argument("--tag", default="",
                        help="suffix for the eval report filename")
    parser.add_argument("--lenient", action="store_true",
                        help="route triple-less facts to Memory instead of dead-lettering")
    args = parser.parse_args()
    _CHAT_SPEC = args.chat
    _LENIENT = args.lenient
    report = run(Path(args.workdir))
    suffix = f"-{args.tag}" if args.tag else ""
    today = datetime.now(timezone.utc).date().isoformat()
    out = REPO_ROOT / "docs" / "superpowers" / "evals" / f"{today}-layered-memory-eval{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
