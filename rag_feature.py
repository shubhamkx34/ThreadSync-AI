import os 
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.tools import tool

# Loading API-Keys :
load_dotenv()
llmModel = ChatGroq(model='openai/gpt-oss-120b')
openrouter_api_key = os.environ["OPENROUTER_API_KEY"]

# Emebedding Model :
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    model="nvidia/nemotron-3-embed-1b:free",
    check_embedding_ctx_length=False # skips that check and sends your plain text straight to OpenRouter
)

# Builds and saves the database :
def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local("faiss_DB")

# Loads the database for searching :
def get_retriever():
    DB_PATH = "faiss_DB"
    # Loads the saved database from your hard drive back into active Python memory, using allow_dangerous_deserialization=True to grant permission to read the local file.
    vector_store = FAISS.load_local(   
            folder_path=DB_PATH,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)