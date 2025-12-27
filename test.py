from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=30
)

print("Sending request...")
response = llm.invoke("Say hello in one short sentence.")
print("Response:", response.content)
