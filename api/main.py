# api/main.py
import sys
from pathlib import Path

# Ensure the repository root is in Python's search path for clean absolute imports
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Project Imports
from utils.logging_config import setup_logging
from api.schemas import ChatRequest, ChatResponse
from agents.react_agent import FinancialAgent

# Initialize logging infrastructure
setup_logging()

# Initialize FastAPI application instance
app = FastAPI(
    title="Financial Document Intelligence API",
    description="Backend API hosting a LangGraph ReAct agent with SQL and tool execution access.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) so your Streamlit app 
# can safely make web calls even if hosted on a different port/domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to explicitly trust your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global holder for the heavy model component agent instance
financial_agent = None

@app.on_event("startup")
def startup_event():
    """
    Executes once when the API server boots up. Loads your heavy LLMs, 
    vector connections, and database agents into memory *once* globally.
    """
    global financial_agent
    print("🚀 Initializing Financial LangGraph ReAct Agent...")
    financial_agent = FinancialAgent()
    print("✅ ReAct Agent fully loaded and listening for queries!")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Simple diagnostic endpoint to verify the server is active."""
    return {"status": "healthy", "agent_loaded": financial_agent is not None}


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(payload: ChatRequest):
    """
    Main execution endpoint. Accepts a JSON body validated by ChatRequest, 
    runs the ReAct execution path, and guarantees a response formatted by ChatResponse.
    """
    if not financial_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent has not finished initializing yet."
        )
    
    try:
        # Execute the LangGraph chain exactly how you did it inside app.py
        agent_response = financial_agent.agent.invoke(
            {"messages": [("user", payload.question)]}
        )
        
        # Parse text out of the final message object in the graph's history channel array
        if "messages" in agent_response and len(agent_response["messages"]) > 0:
            final_answer = agent_response["messages"][-1].content
        else:
            final_answer = str(agent_response)
            
        return ChatResponse(answer=final_answer, status="success")

    except Exception as e:
        # If an engine breaks internally, return a clean HTTP 500 block safely
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Agent Error: {str(e)}"
        )


if __name__ == "__main__":
    # Run server locally on port 8000 when script is executed directly
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)