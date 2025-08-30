# Setting up pydantic Model for Schema validation
from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name : str
    model_provider : str
    system_prompt : str
    messages : List[str]
    allow_search : bool

# Setting up AI Agent from Frontend Request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "meta-llama/llama-4-scout-17b-16e-instruct", "gpt-4o-mini"]

app = FastAPI(title="LangGrapg AI Agent")
@app.post("/chat")

def chat_endpoint(request : RequestState):
    """
    API Endpoint to interact with the chatbot using LangGraph and Search Tools.
    It Dynamically selects the model specified in the request.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error" : "Invalid model name. Kindly select a valid AI model."}
    
    # Create AI AGent and get response from it.
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

#Run app & Explore Swagger UI Docs
if "__name__" == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

