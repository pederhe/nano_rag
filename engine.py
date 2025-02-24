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
    """将聊天历史格式化为提示词"""
    formatted_history = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg["content"]
        
        # 如果不是用户消息，移除<think></think>标签内的思考内容，否则上下文太长了
        if msg["role"] != "user":
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)

def query_ollama(prompt, callback=None, chat_history=None):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
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
    
    # 构建完整提示词
    if chat_history:
        history_text = format_chat_history(chat_history[:-1])  # 不包括最新的用户消息
        system_prompt = f"""你是一个智能助手。以下是之前的对话历史：

{history_text}

请基于以上对话历史回答用户的问题。
"""
        full_prompt = f"{system_prompt}\n\n用户的问题是：\n\n{prompt}\n"
    else:
        full_prompt = prompt
        
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        stream=True,
        callbacks=callbacks,
        temperature=TEMPERATURE
    )
    return llm.invoke(full_prompt)

def query_vllm(message, callback=None, chat_history=None):
    """Query using vLLM"""
    headers = {"Content-Type": "application/json"}
    
    # 构建消息历史
    messages = []
    if chat_history:
        for msg in chat_history[:-1]:  # 不包括最新的用户消息
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
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

def query_llm(message, callback=None, chat_history=None):
    """Select query engine based on configuration"""
    if QUERY_ENGINE == "vllm":
        return query_vllm(message, callback, chat_history)
    else:
        return query_ollama(message, callback, chat_history) 