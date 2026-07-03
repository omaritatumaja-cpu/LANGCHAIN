import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-embeddinggemma-300m",
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    check_embedding_ctx_length=False
)

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(
    retriever,
    name='kb_search',
    description=(
        "Use this tool to search a knowledge base containing statements about a person's preferences. "
        "Use it for any question about what fruits, products, or items the person likes or dislikes. "
        "The knowledge base contains sentences like 'I love apples.' and 'I hate bananas.'"
    )
)

model = init_chat_model(
    model=os.getenv("LM_STUDIO_MODEL"),
    model_provider="openai",
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
)

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant. Your ONLY source of information is the kb_search tool. "
        "For EVERY question, you MUST call the kb_search tool AT LEAST ONCE to find the answer. "
        "Do not rely on your own knowledge. If the tool doesn't return the answer, say you don't know. "
        "Answer the user's question using ONLY the information retrieved from the tool."
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result["messages"][-1].content)
