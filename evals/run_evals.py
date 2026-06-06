"""
evals/run_evals.py

LangSmith evaluation runner.
Scores the agent on:
  1. Correctness  — LLM-as-judge (0.0–1.0)
  2. Tool selection — did it call the expected tool? (0 or 1)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "config" / ".env")

from langsmith import Client
from langsmith.evaluation import evaluate

GOLDEN_PATH  = ROOT / "evals" / "golden_qa.json"
DATASET_NAME = "business-health-golden-qa-v2"


def load_golden_qa():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def ensure_dataset(client: Client) -> str:
    golden   = load_golden_qa()
    datasets = {d.name for d in client.list_datasets()}

    if DATASET_NAME not in datasets:
        print(f"📦 Creating LangSmith dataset '{DATASET_NAME}'...")
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        for qa in golden:
            client.create_example(
                inputs={"question": qa["question"]},
                outputs={"golden_answer": qa["golden_answer"], "expected_tool": qa["expected_tool"]},
                dataset_id=dataset.id,
            )
        print(f"  ✓ {len(golden)} examples created\n")
    else:
        print(f"✓ Dataset '{DATASET_NAME}' already exists\n")

    return DATASET_NAME


def make_agent_target():
    from agents.business_health_agent import build_agent
    agent = build_agent()

    def target(inputs: dict) -> dict:
        result     = agent.invoke({"input": inputs["question"]})
        tools_used = [a.tool for a, _ in result.get("intermediate_steps", [])]
        return {"answer": result["output"], "tools_used": tools_used}

    return target


def tool_selection_evaluator(run, example) -> dict:
    expected   = example.outputs.get("expected_tool", "")
    tools_used = run.outputs.get("tools_used", []) if run.outputs else []
    score      = 1.0 if expected in tools_used else 0.0
    return {
        "key":     "tool_selection",
        "score":   score,
        "comment": f"Expected '{expected}', called: {tools_used}",
    }


def make_correctness_evaluator():
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI
        judge = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0,
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        judge = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(model="gpt-4o", temperature=0)

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analysis evaluator. Score the agent answer vs the golden criteria.
Score 0.0–1.0:
  1.0 = All key facts present and correct
  0.75 = Mostly correct, minor gaps
  0.5 = Key insight present but significant gaps
  0.25 = Barely relevant
  0.0 = Wrong or irrelevant

Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""),
        ("human", "Question: {question}\n\nGolden criteria: {golden_answer}\n\nAgent answer: {agent_answer}"),
    ])

    def evaluator(run, example) -> dict:
        try:
            resp    = (prompt | judge).invoke({
                "question":      example.inputs.get("question", ""),
                "golden_answer": example.outputs.get("golden_answer", ""),
                "agent_answer":  run.outputs.get("answer", "") if run.outputs else "",
            })
            content = resp.content.strip().replace("```json", "").replace("```", "").strip()
            parsed  = json.loads(content)
            return {"key": "correctness", "score": float(parsed.get("score", 0)), "comment": parsed.get("reason", "")}
        except Exception as e:
            return {"key": "correctness", "score": 0.0, "comment": f"Evaluator error: {e}"}

    return evaluator


def run_evals(prefix: str = "business-health"):
    client    = Client()
    dataset   = ensure_dataset(client)
    target    = make_agent_target()
    evaluator = make_correctness_evaluator()

    print(f"🚀 Running evals (prefix: '{prefix}')...\n")

    results = evaluate(
        target,
        data=dataset,
        evaluators=[evaluator, tool_selection_evaluator],
        experiment_prefix=prefix,
        metadata={"version": "2.0"},
        max_concurrency=1,
    )

    # Summary
    c_scores, t_scores = [], []
    print("\n" + "═" * 60)
    print("  📊  EVALUATION RESULTS")
    print("═" * 60)

    for r in results._results:
        q   = r["example"].inputs.get("question", "?")[:55]
        fb  = {f.key: f.score for f in r.get("evaluation_results", {}).get("results", [])}
        c   = fb.get("correctness", 0.0)
        t   = fb.get("tool_selection", 0.0)
        c_scores.append(c)
        t_scores.append(t)
        print(f"  {'✅' if c >= 0.75 else '⚠️' if c >= 0.5 else '❌'} {q}...")
        print(f"     Correctness: {c:.2f}   Tool: {'✅' if t == 1.0 else '❌'}")

    avg_c = sum(c_scores) / len(c_scores) if c_scores else 0
    avg_t = sum(t_scores) / len(t_scores) if t_scores else 0

    print("\n" + "─" * 60)
    print(f"  Avg correctness:    {avg_c:.2f} / 1.00")
    print(f"  Avg tool selection: {avg_t:.2f} / 1.00")
    print(f"\n  🔗 https://smith.langchain.com/\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="business-health")
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  LANGSMITH_API_KEY not set — see config/.env")
        sys.exit(1)

    run_evals(prefix=args.prefix)
