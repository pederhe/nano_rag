import os
import re
import sys
from queue import Queue
from threading import Thread

import chromadb
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_TYPE,
)
from engine import format_chat_history, query_llm

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

def check_embedding_consistency():
    """
    Check if the embedding type used in ChromaDB collection is consistent with configuration
    """
    try:
        # Get all collection names
        collection_names = chroma_client.list_collections()
        
        # Check if target collection exists
        if collection_name in collection_names:
            # Get collection details using get_collection
            col = chroma_client.get_collection(name=collection_name)
            metadata = col.metadata
            if metadata and "embedding_type" in metadata:
                stored_type = metadata["embedding_type"]
                if stored_type != EMBEDDING_TYPE:
                    print(f"""
Error: Embedding type mismatch!
Configured type: {EMBEDDING_TYPE}
Type used in database: {stored_type}

Please do one of the following:
1. Change EMBEDDING_TYPE in config file to "{stored_type}"
2. Delete ChromaDB data directory and recreate (will lose all data)
3. Use a new collection name
""")
                    sys.exit(1)
    except Exception as e:
        print(f"Error checking embedding consistency: {str(e)}")
        sys.exit(1)

def check_huggingface_token():
    """Check if Hugging Face token is set when using nomic"""
    if EMBEDDING_TYPE == "nomic" and not os.getenv('HUGGINGFACEHUB_API_TOKEN'):
        print("""
Error: Using nomic embedding requires setting HUGGINGFACEHUB_API_TOKEN environment variable

Please set the environment variable and try again:
Linux/Mac: export HUGGINGFACEHUB_API_TOKEN="your_token_here"
Windows: set HUGGINGFACEHUB_API_TOKEN=your_token_here
""")
        sys.exit(1)

# Define a custom embedding function for ChromaDB using Nomic embeddings
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB supporting multiple embedding backends.
    """
    def __init__(self):
        if EMBEDDING_TYPE == "nomic":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v2-moe",
                encode_kwargs={'normalize_embeddings': True},
                model_kwargs={'trust_remote_code': True}  # Add this parameter to trust remote code
            )
        else:  # default to ollama
            self.embeddings = OllamaEmbeddings(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_HOST
            )

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        try:
            embeddings = self.embeddings.embed_documents(input)
            return embeddings
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            raise

# Check before initializing embedding
check_huggingface_token()
embedding = ChromaDBEmbeddingFunction()

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo"

# Check embedding consistency before creating/getting collection
check_embedding_consistency()

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={
        "description": f"A collection for RAG Demo using {EMBEDDING_TYPE} embeddings",
        "embedding_type": EMBEDDING_TYPE  # Store embedding type in metadata
    },
    embedding_function=embedding  # Use the custom embedding function
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    try:
        collection.add(
            documents=documents,
            ids=ids
        )
    except Exception as e:
        print(f"Failed to add documents: {str(e)}")
        raise

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=3):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        tuple: (documents, metadatas)
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results["documents"], results["metadatas"]
    except Exception as e:
        print(f"Query error: {str(e)}")
        raise

def generate_summary(content):
    """
    Generate document summary using the configured engine
    
    Args:
        content (str): Document content
    
    Returns:
        str: Generated summary
    """
    prompt = f"Please summarize the main content in one sentence (single language, maximum 200 characters):\n\n{content[:10000]}"
    # Create a queue to receive responses
    queue = Queue()
    summary = ""

    class SummaryCallback(BaseCallbackHandler):
        def __init__(self, queue):
            self.queue = queue
        
        def on_llm_new_token(self, token, **kwargs):
            self.queue.put(token)
        
        def on_llm_end(self, *args, **kwargs):
            self.queue.put(None)

    callback = SummaryCallback(queue)
    
    # Run query in a new thread
    Thread(target=lambda: query_llm(prompt, callback)).start()
    
    # Collect response
    while True:
        token = queue.get()
        if token is None:
            break
        summary += token

    # Remove <think></think> tags and their content
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL) 
    return summary[:500]  # Limit summary length

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text, callback=None, chat_history=None):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
        callback (function): Optional callback function for streaming output
        chat_history (list): Optional chat history
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = "\n---\n".join(["\n".join(docs) for docs in retrieved_docs]) if retrieved_docs else "No relevant content"

    # Step 2: Build complete prompt with context and chat history
    if len(chat_history) > 0:
        history_text = format_chat_history(chat_history[:-1])
        system_prompt = f"""You are a professional knowledge base assistant who must strictly follow these rules when answering questions:
1. Answer questions based ONLY on the provided knowledge base content (referred to as [Knowledge Base]), do not fabricate information.
2. If the question is outside the scope of the [Knowledge Base], directly respond with "Based on available information, no relevant data was found" without adding anything else.
3. Answers should be concise, accurate, and reference file names or paragraph numbers from the [Knowledge Base].
4. If the user's question is ambiguous, first request clarification of specific requirements.

[Knowledge Base] content:
{context}

Conversation history:
{history_text}"""
        augmented_prompt = f"{system_prompt}\n\nQuestion:\n\n{query_text}\n"
    else:
        system_prompt = f"""You are a professional knowledge base assistant who must strictly follow these rules when answering questions:
1. Answer questions based ONLY on the provided knowledge base content (referred to as [Knowledge Base]), do not fabricate information.
2. If the question is outside the scope of the [Knowledge Base], directly respond with "Based on available information, no relevant data was found" without adding anything else.
3. Answers should be concise, accurate, and reference file names or paragraph numbers from the [Knowledge Base].
4. If the user's question is ambiguous, first request clarification of specific requirements.

[Knowledge Base] content:
{context}"""
        augmented_prompt = f"{system_prompt}\n\nQuestion:\n\n{query_text}\n"

    
    if os.getenv('DEBUG'):
        print("\n=== Debug: Full Prompt ===")
        print(augmented_prompt)
        print("=========================\n")
    
    return query_llm(augmented_prompt, callback)
