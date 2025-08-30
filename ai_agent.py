import os
import getpass
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Setting up LLMs and Tools

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.ai import AIMessage

OPENAI_LLM = ChatOpenAI(model="gpt-4o-mini")
GROQ_LLM = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")



# Creating AI Agent with search tool Functionality

from langgraph.prebuilt import create_react_agent
# system_prompt = "Act as an AI agent who is smart and friendly."

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == 'GroQ':
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model = llm,
        tools = tools,
        prompt = system_prompt
        )

    # query = "Tell me about trends in crypto market?"
    state = {"messages":query}
    response = agent.invoke(state)
    msgs = response.get("messages")
    ai_message = [message.content for message in msgs if isinstance(message, AIMessage)]
    return ai_message[-1]
