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
    检查ChromaDB中的collection使用的embedding类型是否与配置一致
    """
    try:
        # 获取所有collection名称
        collection_names = chroma_client.list_collections()
        
        # 检查目标collection是否存在
        if collection_name in collection_names:
            # 使用get_collection获取collection详情
            col = chroma_client.get_collection(name=collection_name)
            metadata = col.metadata
            if metadata and "embedding_type" in metadata:
                stored_type = metadata["embedding_type"]
                if stored_type != EMBEDDING_TYPE:
                    print(f"""
错误: Embedding类型不匹配!
配置的类型: {EMBEDDING_TYPE}
数据库使用的类型: {stored_type}

请执行以下操作之一:
1. 修改配置文件中的EMBEDDING_TYPE为"{stored_type}"
2. 删除ChromaDB数据目录并重新创建 (将丢失所有数据)
3. 使用新的collection名称
""")
                    sys.exit(1)
    except Exception as e:
        print(f"检查Embedding一致性时发生错误: {str(e)}")
        sys.exit(1)

def check_huggingface_token():
    """检查使用 nomic 时是否设置了 Hugging Face token"""
    if EMBEDDING_TYPE == "nomic" and not os.getenv('HUGGINGFACEHUB_API_TOKEN'):
        print("""
错误: 使用 nomic embedding 需要设置 HUGGINGFACEHUB_API_TOKEN 环境变量

请设置环境变量后重试:
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
                model_kwargs={'trust_remote_code': True}  # 添加此参数以信任远程代码
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

# 在初始化 embedding 前检查
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
    prompt = f"请用一句话总结主要内容(单一语言,最多200个字):\n\n{content[:10000]}"
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
    context = "\n---\n".join(["\n".join(docs) for docs in retrieved_docs]) if retrieved_docs else "没有相关内容"

    # Step 2: 构建包含上下文和聊天历史的完整提示词
    if len(chat_history) > 0:
        history_text = format_chat_history(chat_history[:-1])
        system_prompt = f"""你是一个专业的知识库助手，需要严格按照以下规则回答问题：
1. 仅基于用户提供的知识库内容（以下用[知识库]代指）回答问题，禁止编造信息。
2. 如果问题超出[知识库]范围，直接回答"根据现有资料，暂未找到相关信息"，不要加入其他内容。
3. 回答需简洁、准确，并引用[知识库]中的文件名或段落编号。
4. 如果用户的问题模糊，请先请求澄清具体需求。

[知识库]内容：
{context}

对话历史：
{history_text}"""
        augmented_prompt = f"{system_prompt}\n\n问题：\n\n{query_text}\n"
    else:
        system_prompt = f"""你是一个专业的知识库助手，需要严格按照以下规则回答问题：
1. 仅基于用户提供的知识库内容（以下用[知识库]代指）回答问题，禁止编造信息。
2. 如果问题超出[知识库]范围，直接回答"根据现有资料，暂未找到相关信息"，不要加入其他内容。
3. 回答需简洁、准确，并引用[知识库]中的文件名或段落编号。
4. 如果用户的问题模糊，请先请求澄清具体需求。

[知识库]内容：
{context}"""
        augmented_prompt = f"{system_prompt}\n\n问题：\n\n{query_text}\n"

    
    if os.getenv('DEBUG'):
        print("\n=== Debug: Full Prompt ===")
        print(augmented_prompt)
        print("=========================\n")
    
    return query_llm(augmented_prompt, callback)
