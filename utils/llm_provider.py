import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct",
        temperature=0,

        # 🔑 OpenRouter API key (stored as OPENAI_API_KEY)
        api_key=os.getenv("OPENAI_API_KEY"),

        # 🌐 THIS IS THE MOST IMPORTANT LINE
        base_url="https://openrouter.ai/api/v1",

        # 🧾 Optional but recommended headers
        default_headers={
            "HTTP-Referer": "http://localhost:8501",  # or your project URL
            "X-Title": "Document-QA-App"
        }
    )
