"""
agents/business_health_agent.py

Business Health Agent — LangChain ReAct agent.

Features:
  - Auto-detects LLM: Azure OpenAI → Anthropic Claude → OpenAI
  - Reads from Kaggle CSV or mock JSON (handled by data/loader.py)
  - Conversation memory (last 6 turns)
  - Optional streaming output (--stream flag)
  - LangSmith tracing via LANGSMITH_API_KEY env var
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── LLM factory ──────────────────────────────────────────────────────────────
def _build_llm(streaming: bool = False):
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0,
            streaming=streaming,
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=0,
            streaming=streaming,
        )
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            streaming=streaming,
        )
    else:
        raise EnvironmentError(
            "No LLM API key found.\n"
            "Set one of: AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
        )


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable financial analyst assistant for a small business.
You help the owner understand their financial health by answering questions clearly and concisely.
The data comes from a real bookkeeping dataset — use it faithfully.

You have access to these tools:
{tools}

STRICT FORMAT — follow exactly:

Question: the input question you must answer
Thought: which tool to use and why
Action: one of [{tool_names}]
Action Input: the input to the action (plain string)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: a clear, professional answer in plain English. Include key numbers,
highlight risks, and give context. Use $ currency formatting.

Rules:
- Always call a tool — never guess financial figures
- Be specific with numbers; round to 2 decimal places
- Flag risks (overdue receivables, low cash, negative cash flow)
- For comparisons, call the tool for each period separately
- Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ── Agent factory ─────────────────────────────────────────────────────────────
def build_agent(streaming: bool = False):
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.prompts import PromptTemplate
    from tools.financial_tools import ALL_TOOLS

    llm    = _build_llm(streaming=streaming)
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "chat_history"],
        partial_variables={
            "tools":      "\n".join(f"{t.name}: {t.description}" for t in ALL_TOOLS),
            "tool_names": ", ".join(t.name for t in ALL_TOOLS),
        },
        template=SYSTEM_PROMPT,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=6, return_messages=False,
    )
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ── Streaming helper ──────────────────────────────────────────────────────────
def stream_response(agent, question: str):
    for chunk in agent.stream({"input": question}):
        if "output" in chunk:
            yield chunk["output"]
        elif isinstance(chunk, str):
            yield chunk


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / ".env")

    parser = argparse.ArgumentParser(description="Business Health Agent")
    parser.add_argument("--stream",    action="store_true", help="Enable streaming output")
    parser.add_argument("--question", "-q", type=str,       help="Single question then exit")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  💼  Business Health Agent")
    print("═" * 60)

    llm_name = (
        "Azure OpenAI"  if os.getenv("AZURE_OPENAI_API_KEY") else
        "Claude Sonnet" if os.getenv("ANTHROPIC_API_KEY")    else
        "OpenAI GPT-4o"
    )
    langsmith = "✓ active" if os.getenv("LANGSMITH_API_KEY") else "✗ not configured"
    print(f"  LLM: {llm_name}   |   LangSmith: {langsmith}")
    print("  Type 'exit' to quit\n")

    agent = build_agent(streaming=args.stream)

    if args.question:
        result = agent.invoke({"input": args.question})
        print(f"\n📊 Answer:\n{result['output']}\n")
        sys.exit(0)

    while True:
        try:
            question = input("🔍 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("exit", "quit", "q", ""):
            print("Goodbye!")
            break

        try:
            if args.stream:
                print("\n📊 Answer: ", end="", flush=True)
                for token in stream_response(agent, question):
                    print(token, end="", flush=True)
                print("\n")
            else:
                result = agent.invoke({"input": question})
                print(f"\n📊 Answer:\n{result['output']}\n")
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")
