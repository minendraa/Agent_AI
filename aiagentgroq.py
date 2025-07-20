import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from functools import lru_cache
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun 
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType 

load_dotenv()

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    agent_steps: List[str]
    final_answer: str


app = FastAPI()


@lru_cache(maxsize=1)
def get_agent():
    groq_api_key = os.getenv("GROQ_API")
    if not groq_api_key:
        raise RuntimeError("GROQ_API not found. Put it in your .env file or as an environment variable.")

    # ✅ Correct model name (must match one available in your Groq dashboard)
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",  # ✅ Ensure this is available in your account
    )

    # ✅ DuckDuckGoSearchRun import was updated
    search_tool = DuckDuckGoSearchRun()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [search_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ✅ Proper enum
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    agent = get_agent()
    try:
        result = await agent.ainvoke({"input": request.question})

        # Extract steps from the result (not always available)
        intermediate_steps = result.get("intermediate_steps", [])
        final_answer = result.get("output", "")

        return AskResponse(
            question=request.question,
            agent_steps=[str(step) for step in intermediate_steps],
            final_answer=final_answer,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
