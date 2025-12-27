from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Returns FREE local embeddings.
    Works for both Gemini and OpenAI LLMs.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
