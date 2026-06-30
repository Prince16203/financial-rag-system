"""
agents/react_agent.py

ReAct (Reasoning + Acting) agent using LangGraph.
Orchestrates tools: query_metrics_db, calculator, search_documents.

The agent decides which tool to use based on the question, iterates
until it has a final answer, then returns the result.

Usage:
    from agents.react_agent import ReactAgent

    agent = ReactAgent()
    result = agent.run("What is Apple's FICC revenue for 2024?")
    print(result)
"""

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

import config
from agents.tools import TOOLS, query_metrics_db, calculator, search_documents
from agents.sql_agent import SQLAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent State — what flows through the graph
# ---------------------------------------------------------------------------

class AgentState:
    """State passed through LangGraph nodes."""
    
    def __init__(self, messages: list[BaseMessage] = None):
        self.messages = messages or []
        self.final_answer = None
        self.tool_calls = 0
        self.max_iterations = 10


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReactAgent:
    """
    Financial question-answering agent using ReAct pattern.
    
    Flow:
        1. User asks a question
        2. Agent reads question + available tools
        3. Agent decides: use a tool OR return final answer
        4. If tool: execute it, add result to messages, loop
        5. If final: format and return answer
    
    Tools available:
        - query_metrics_db: Get structured metrics from SQLite
        - calculator: Do arithmetic on numbers
        - search_documents: Fall back to RAG for context
    """
    
    def __init__(
        self,
        groq_model: str = config.GROQ_MODEL,
        groq_api_key: Optional[str] = None,
        max_iterations: int = 10,
    ):
        self.groq_model = groq_model
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        self.max_iterations = max_iterations
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        
        # LangChain's ChatOpenAI can call Groq via OpenAI-compatible endpoint
        # (groq-sdk handles this natively, but LangGraph prefers ChatOpenAI)
        self.llm = ChatOpenAI(
            model=self.groq_model,
            api_key=self.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
        )
        
        self.tools = TOOLS
        self.sql_agent = SQLAgent()
        
        # Bind tools to LLM so it can call them
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(
            f"ReactAgent ready | "
            f"model={self.groq_model} | "
            f"tools={len(self.tools)} | "
            f"max_iterations={max_iterations}"
        )
    
    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    
    def run(self, question: str) -> str:
        """
        Run the agent on a question.
        
        Args:
            question: Natural language financial question
        
        Returns:
            Final answer string
        """
        logger.info(f"Agent.run: '{question[:80]}'")
        
        messages = [HumanMessage(content=question)]
        
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Call LLM with tools bound
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if LLM wants to stop
            if not response.tool_calls:
                # No more tool calls — this is the final answer
                logger.info(f"Agent finished | iterations={iteration + 1}")
                return response.content
            
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                
                logger.debug(f"Calling tool: {tool_name} with {tool_input}")
                
                # Find and execute the tool
                result = self._execute_tool(tool_name, tool_input)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result,
                })
            
            # Add tool results to messages
            for tool_result in tool_results:
                messages.append(
                    ToolMessage(
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_use_id"],
                    )
                )
        
        # Max iterations reached
        logger.warning(f"Agent hit max iterations ({self.max_iterations})")
        return (
            "I reached the maximum number of steps. "
            "The question may be too complex or require data not in the system."
        )
    
    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a single tool by name.
        
        Args:
            tool_name: Name of the tool ("query_metrics_db", "calculator", etc.)
            tool_input: Arguments dict
        
        Returns:
            Tool output as string
        """
        try:
            if tool_name == "query_metrics_db":
                return query_metrics_db.invoke(tool_input.get("query", ""))
            
            elif tool_name == "calculator":
                return calculator.invoke(tool_input.get("expression", ""))
            
            elif tool_name == "search_documents":
                return search_documents.invoke(tool_input.get("question", ""))
            
            else:
                return f"Unknown tool: {tool_name}"
        
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return f"Error executing {tool_name}: {e}"
    
    # ------------------------------------------------------------------
    # Utility: multi-question mode
    # ------------------------------------------------------------------
    
    def ask_multiple(self, questions: list[str]) -> dict[str, str]:
        """
        Run the agent on multiple questions (useful for benchmarking).
        
        Args:
            questions: List of questions
        
        Returns:
            Dict mapping each question to its answer
        """
        results = {}
        for q in questions:
            results[q] = self.run(q)
        return results


# ---------------------------------------------------------------------------
# LangGraph Version (Alternative)
# ---------------------------------------------------------------------------
# If you prefer a more explicit state-based graph (optional):
# from langgraph.graph import StateGraph
# from typing import TypedDict, Annotated
# import operator
#
# class GraphState(TypedDict):
#     messages: Annotated[list[BaseMessage], operator.add]
#
# def agent_node(state: GraphState):
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}
#
# def should_continue(state: GraphState):
#     last = state["messages"][-1]
#     return "end" if not last.tool_calls else "continue"
#
# graph = StateGraph(GraphState)
# graph.add_node("agent", agent_node)
# graph.add_node("tools", ToolNode(TOOLS))
# graph.add_edge("tools", "agent")
# graph.add_conditional_edges("agent", should_continue, ...)
# graph.set_entry_point("agent")
# graph.set_finish_point("agent")
#
# compiled = graph.compile()
# result = compiled.invoke({"messages": [HumanMessage(content=question)]})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()
    
    agent = ReactAgent()
    
    questions = [
        "What is Apple's revenue for 2024?",
        "Calculate 391.04 minus 365.78",
        "What are Goldman Sachs' main risk factors?",
    ] if len(sys.argv) == 1 else sys.argv[1:]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        answer = agent.run(q)
        print(f"Answer:\n{answer}")