from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(documents):
    """
    Splits documents into chunks for embeddings.

    Returns:
    - list of chunked LangChain Documents
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks
