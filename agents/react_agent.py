"""
agents/react_agent.py

ReAct agent using LangGraph that orchestrates all three tools:
- query_metrics_db: for structured financial numbers
- calculator: for arithmetic
- search_documents: for qualitative text questions

ReAct loop:
    Thought -> Action -> Observation -> Thought -> ... -> Final Answer

Usage:
    from agents.react_agent import FinancialAgent

    agent = FinancialAgent()
    result = agent.run("What was Apple revenue in 2024 and how does it compare to 2023?")
    print(result["output"])
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

import config
from agents.tools import TOOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """You are a financial document analyst with access to tools.

You have three tools available:
1. query_metrics_db — use for specific financial numbers (revenue, net income, FICC revenue etc.)
2. calculator — use for any arithmetic (percentage change, ratios, differences)
3. search_documents — use for qualitative questions (risk factors, business strategy, descriptions)

IMPORTANT RULES:
- query_metrics_db only accepts ONE company and ONE metric at a time. 
  For comparisons, call it separately for each company.
- calculator only accepts numeric expressions with actual numbers — 
  never use placeholder names like result1 or result2. 
  Read the actual numeric values from previous tool outputs and substitute them directly.
- If query_metrics_db returns no results for a specific year, try without 
  specifying the year, or try search_documents as a fallback.
- Always state the actual year used if it differs from what was asked.

Always follow this process:
- Think about what information you need
- Use query_metrics_db FIRST for any numeric financial question, one company per call
- Once you have two numbers, use calculator with the literal numbers, not variable names
- Use search_documents for context and narrative questions
- Always cite the source PDF and page in your final answer
"""


class FinancialAgent:
    """
    ReAct agent for financial document analysis.

    Combines SQL database queries, calculations, and document
    search to answer complex financial questions.

    Args:
        model:       Groq model to use for reasoning.
        temperature: LLM temperature. Keep low for factual answers.
        verbose:     If True prints each reasoning step.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        model = model or config.GROQ_MODEL

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        # LangChain ChatGroq wrapper — required for LangGraph
        self.llm = ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
        )

        # Create ReAct agent using LangGraph prebuilt
        # This handles the Thought/Action/Observation loop automatically
        self.agent = create_react_agent(
            model=self.llm,
            tools=TOOLS,
            prompt=SYSTEM_MESSAGE,
        )

        self.verbose = verbose
        logger.info(f"FinancialAgent ready | model={model} | tools={[t.name for t in TOOLS]}")

    def run(self, question: str) -> dict:
        """
        Run the ReAct agent on a question.

        The agent will:
        1. Reason about which tools to use
        2. Call tools in sequence
        3. Observe results
        4. Produce a final answer

        Args:
            question: Natural language financial question.

        Returns:
            Dict with 'output' (final answer string) and
            'steps' (list of tool calls made).
        """
        logger.info(f"Agent question: '{question[:80]}'")

        messages = [HumanMessage(content=question)]

        try:
            result = self.agent.invoke({"messages": messages})

            # Extract final answer from last message
            final_message = result["messages"][-1]
            output = final_message.content

            # Extract tool calls made during reasoning
            steps = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        steps.append({
                            "tool": tc["name"],
                            "input": tc["args"],
                        })

            if self.verbose:
                self._print_steps(question, steps, output)

            return {
                "output": output,
                "steps":  steps,
                "messages": result["messages"],
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "output": f"Agent encountered an error: {e}",
                "steps":  [],
                "messages": [],
            }

    def _print_steps(self, question: str, steps: list, output: str) -> None:
        """Print reasoning steps in readable format."""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        if steps:
            print(f"\nTools used ({len(steps)} calls):")
            for i, step in enumerate(steps, 1):
                print(f"  [{i}] {step['tool']}")
                input_str = str(step['input'])[:100]
                print(f"      Input: {input_str}")

        print(f"\nFinal Answer:")
        print(output)
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Install langgraph if not already installed
    try:
        import langgraph
    except ImportError:
        print("Installing langgraph...")
        import subprocess
        subprocess.run(["pip", "install", "langgraph", "langchain-groq"])

    questions = [
        "What was Apple's total revenue in 2024?",
        "What are the main risk factors disclosed by Goldman Sachs?",
        "Calculate the percentage: if Apple revenue was 391.04 billion in 2024, what is 10% of that?",
    ]

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else questions[0]

    agent = FinancialAgent()
    result = agent.run(question)