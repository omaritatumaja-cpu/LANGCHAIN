import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads .env and loads into environment variables

llm = ChatOpenAI(
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    model=os.getenv("LM_STUDIO_MODEL"),
    temperature=0.1,
    max_tokens=256
)

response = llm.invoke("Hello, are you working?")
print(response.content)
