from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings


def create_vectorstore(chunks):
    """
    Creates a FAISS vector store from document chunks.
    Handles empty chunks safely.
    """

    if not chunks:
        raise ValueError("No valid text chunks found in the document.")

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb
