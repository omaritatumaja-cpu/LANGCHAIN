from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from dotenv import load_dotenv
import requests
from dataclasses import dataclass
import os

# or just don't set it, false is current default
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"


load_dotenv()


# ---------- Tools ----------

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime) -> str:
    user_id = runtime.context.user_id
    match user_id:
        case '1':
            return 'Tokyo'
        case '2':
            return 'Paris'
        case 'ABC123':
            return 'Berlin'
        case _:
            return 'Unknown'


# ---------- Schemas ----------

@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


# ---------- Model (local via LM Studio) ----------

model = init_chat_model(
    model=os.getenv("LM_STUDIO_MODEL"),
    model_provider="openai",
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    temperature=0.3,
    max_tokens=256,
)

# ---------- Memory ----------

checkpointer = InMemorySaver()

# ---------- Agent ----------

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt='You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.',
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}

# ---------- Call 1: city explicitly mentioned ----------

response = agent.invoke(
    {'messages': [
        {'role': 'user', 'content': 'What is the weather like ?'}]},
    config=config,
    context=Context(user_id='ABC123')
)

print('--- Call 1 ---')
print(response['messages'][-1].content)
print(response['structured_response'])

# ---------- Call 2: no city mentioned, agent must use locate_user + memory ----------

response = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What is the weather like?'}]},
    config=config,
    context=Context(user_id='ABC123')
)

print('--- Call 2 ---')
print(response['messages'][-1].content)
print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)
