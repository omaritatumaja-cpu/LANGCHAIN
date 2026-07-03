import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    model=os.getenv("LM_STUDIO_MODEL"),
    temperature=0.1,
    max_tokens=256,
)


@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt='You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.'
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in Rumonge now?'}
    ]
})


print(response['messages'][-1].content)
