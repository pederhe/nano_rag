import json
import re
import requests
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM

from config import (
    OLLAMA_MODEL,
    QUERY_ENGINE,
    VLLM_HOST,
    VLLM_MODEL,
    TEMPERATURE
)

def format_chat_history(history):
    """Format chat history into prompt"""
    formatted_history = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        
        # If not a user message, remove thinking content in <think></think> tags to reduce context length
        if msg["role"] != "user":
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)

def query_ollama(system_prompt, message, callback=None, chat_history=None):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        system_prompt (str): The system prompt for Ollama.
        message (str): The input message for Ollama.
        callback (function): Optional callback function for streaming output
        chat_history (list): Optional chat history
    
    Returns:
        str: The response from Ollama.
    """
    callbacks = []
    if callback:
        callbacks.append(callback)
    else:
        callbacks.append(StreamingStdOutCallbackHandler())
    
    # Build complete prompt
    if chat_history:
        history_text = format_chat_history(chat_history[:-1])  # Excluding the latest user message
        full_prompt = f"""{system_prompt}\n\nHere is the previous conversation history:

{history_text}

Please answer the user's question based on the above conversation history.\n\nUser's question is:\n\n{message}\n
"""
    else:
        full_prompt = f"{system_prompt}\n\n{message}\n"
        
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        stream=True,
        callbacks=callbacks,
        temperature=TEMPERATURE
    )
    return llm.invoke(full_prompt)

def query_vllm(system_prompt, message, callback=None, chat_history=None):
    """Query using vLLM"""
    headers = {"Content-Type": "application/json"}
    
    # Build message history
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    if chat_history:
        for msg in chat_history[:-1]:  # Excluding the latest user message
            if msg["role"] != "user":
                content = re.sub(r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL)
            messages.append({
                "role": msg["role"],
                "content": content
            })
    messages.append({"role": "user", "content": message})
    
    data = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": True
    }

    response = requests.post(f"{VLLM_HOST}/v1/chat/completions", 
                           headers=headers,
                           json=data,
                           stream=True)

    for line in response.iter_lines():
        if line:
            json_str = line.decode('utf-8').removeprefix('data: ')
            if json_str.strip() == '[DONE]':
                break
            try:
                chunk = json.loads(json_str)
                if chunk.get('choices'):
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content and callback:
                        callback.on_llm_new_token(content)
            except json.JSONDecodeError:
                continue

    if callback:
        callback.on_llm_end()

def query_llm(system_prompt, message, callback=None, chat_history=None):
    """Select query engine based on configuration"""
    if QUERY_ENGINE == "vllm":
        return query_vllm(system_prompt, message, callback, chat_history)
    else:
        return query_ollama(system_prompt, message, callback, chat_history) 