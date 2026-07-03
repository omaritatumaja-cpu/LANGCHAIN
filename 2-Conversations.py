import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model=os.getenv("LM_STUDIO_MODEL"),
    model_provider="openai",
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    temperature=0.1
)

conversation = [
    SystemMessage(
        'You are a helpful assistant for questions regarding programming'),
    HumanMessage('What is Python?'),
    AIMessage('Python is an interpreted programming language.'),
    HumanMessage('When was it released?')
]

response = model.invoke(conversation)


print(response.content)
