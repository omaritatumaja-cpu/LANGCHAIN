import os
import base64
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model=os.getenv("LM_STUDIO_MODEL"),
    model_provider="openai",
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
)

# Download the image and convert it to base64
image_response = requests.get(
    'https://neuralnine.com/wp-content/uploads/2025/04/neuralnine_logo_transparent-1536x740.png'
)
image_data = base64.b64encode(image_response.content).decode('utf-8')

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image', 'url': f'data:image/png;base64,{image_data}'}
    ]
}

response = model.invoke([message])

print(response.content)
