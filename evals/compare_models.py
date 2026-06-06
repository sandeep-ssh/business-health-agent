"""
evals/compare_models.py  — Stretch goal: compare two models side by side in LangSmith.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "config" / ".env")

from langsmith import Client
from langsmith.evaluation import evaluate
from evals.run_evals import (
    ensure_dataset, make_correctness_evaluator, tool_selection_evaluator
)

MODELS = [
    {
        "label":  "Claude Sonnet 4",
        "prefix": "compare-claude-sonnet4",
        "env":    {"ANTHROPIC_MODEL": "claude-sonnet-4-20250514"},
    },
    {
        "label":  "Claude Haiku 4.5",
        "prefix": "compare-claude-haiku",
        "env":    {"ANTHROPIC_MODEL": "claude-haiku-4-5-20251001"},
    },
]


def _target_for(env_override: dict):
    import os
    orig = {k: os.environ.get(k, "") for k in env_override}
    for k, v in env_override.items():
        os.environ[k] = v

    from agents.business_health_agent import build_agent
    agent = build_agent()

    for k, v in orig.items():
        if v: os.environ[k] = v
        elif k in os.environ: del os.environ[k]

    def target(inputs):
        r = agent.invoke({"input": inputs["question"]})
        return {"answer": r["output"], "tools_used": [a.tool for a, _ in r.get("intermediate_steps", [])]}
    return target


def compare_models():
    client    = Client()
    dataset   = ensure_dataset(client)
    evaluator = make_correctness_evaluator()
    scores    = {}

    for m in MODELS:
        print(f"\n🤖 Evaluating: {m['label']}")
        results = evaluate(
            _target_for(m["env"]),
            data=dataset,
            evaluators=[evaluator, tool_selection_evaluator],
            experiment_prefix=m["prefix"],
            metadata={"model": m["label"]},
            max_concurrency=1,
        )
        c, t = [], []
        for r in results._results:
            fb = {f.key: f.score for f in r.get("evaluation_results", {}).get("results", [])}
            c.append(fb.get("correctness", 0.0))
            t.append(fb.get("tool_selection", 0.0))
        scores[m["label"]] = {"correctness": sum(c)/len(c) if c else 0, "tool_selection": sum(t)/len(t) if t else 0}

    print("\n" + "═" * 58)
    print("  🏆  MODEL COMPARISON")
    print("═" * 58)
    print(f"  {'Model':<26} {'Correctness':>12} {'Tool Sel.':>10}")
    print("  " + "─" * 54)
    for label, s in scores.items():
        print(f"  {label:<26} {s['correctness']:>11.2f}  {s['tool_selection']:>9.2f}")
    print("═" * 58)
    print("\n  Full details: https://smith.langchain.com/\n")


if __name__ == "__main__":
    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  LANGSMITH_API_KEY not set")
        sys.exit(1)
    compare_models()
