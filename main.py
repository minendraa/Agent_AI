# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

# Set up Together.ai-compatible ChatOpenAI
llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # You can change this
    temperature=0.1,
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1",
)

# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun()

# Define tools
tools = [
    Tool(
        name="duckduckgo_search",
        func=search_tool.run,
        description="Use this tool to search the web for current information or facts."
    )
]

# LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.post("/ask", response_model=QueryResponse)
async def ask(query: QueryRequest):
    try:
        result = agent.run(query.query)
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
