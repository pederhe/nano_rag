import json
import math
import secrets
from queue import Queue
from threading import Thread
import re
import time

from flask import Flask, render_template, request, Response, stream_with_context, redirect, url_for, session
from flask_wtf.csrf import CSRFProtect
from langchain.callbacks.base import BaseCallbackHandler

from document_processor import DocumentProcessor
from rag import rag_pipeline, query_llm

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set secret key for session
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit upload file size to 1MB
csrf = CSRFProtect(app)  # Add CSRF protection

doc_processor = DocumentProcessor()

# Global dictionary to store chat histories
chat_histories = {}

def get_history():
    # Use csrf_token as session identifier
    session_id = session.get('csrf_token')
    if not session_id:
        session_id = secrets.token_hex(16)
        session['csrf_token'] = session_id
    
    # Initialize chat history for new sessions
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    return chat_histories[session_id]

def format_think_content(content):
    """Convert <think></think> tags to HTML think area"""
    if '<think>' not in content:
        return content
    
    parts = content.split('<think>')
    result = parts[0]
    
    for part in parts[1:]:
        if '</think>' in part:
            think_parts = part.split('</think>')
            # Extract thinking time
            think_content = think_parts[0]
            time_match = re.search(r'took (\d+) seconds$', think_content)
            if time_match:
                think_time = f'(took {time_match.group(1)} seconds)'
                # Remove time information from content
                think_content = think_content[:-(len(time_match.group(0)))]
            else:
                think_time = ''
            
            result += f'<div class="think-content"><div class="think-header">Deep thinking completed {think_time}</div><div class="think-body">{think_content}</div></div>'
            if len(think_parts) > 1:
                result += think_parts[1]
    
    return result

class StreamingCallback(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.has_think_start = False
        self.response_text = ""
        self.think_start_time = None
        
    def on_llm_new_token(self, token, **kwargs):
        self.response_text += token
        if "<think>" in token:
            self.has_think_start = True
            self.think_start_time = time.time()
        elif "</think>" in token and self.has_think_start:
            # Calculate thinking time
            think_time = int(time.time() - self.think_start_time)
            # Insert time information before closing tag
            token = f'took {think_time} seconds</think>'
            self.think_start_time = None
        self.queue.put(token)
        
    def on_llm_end(self, *args, **kwargs):
        # If there's an opening tag but no closing tag, add closing tag
        if self.has_think_start and "</think>" not in self.response_text:
            self.queue.put("</think>")
        self.queue.put(None)
        
    def on_llm_error(self, *args, **kwargs):
        # If there's an opening tag but no closing tag, add closing tag
        if self.has_think_start and "</think>" not in self.response_text:
            self.queue.put("</think>")
        self.queue.put(None)

@app.route('/')
def index():
    # Display chat page by default
    return render_template('index.html', 
                         active_page='chat',
                         content_url=url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    chat_history = get_history()
    if len(chat_history) == 0:
        chat_history.append(
            {
                "role": "assistant",
                "content": "Hi~ I'm Shiyi, your intelligent assistant"
            }
        )
    # Process think tags in historical messages
    for message in chat_history:
        if message['role'] == 'assistant':
            message['content'] = format_think_content(message['content'])

    return render_template('chat.html', history=chat_history)

@app.route('/query', methods=['POST'])
def query():
    user_message = request.json.get('message', '')
    use_rag = request.json.get('use_rag', False)
    
    # Get chat history for current session
    chat_history = get_history()
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    
    def generate():
        queue = Queue()
        callback = StreamingCallback(queue)
        
        response_text = ""
        def process_query():
            if use_rag:
                rag_pipeline(user_message, callback, chat_history)
            else:
                query_llm(user_message, callback, chat_history)
        
        Thread(target=process_query).start()
        
        while True:
            token = queue.get()
            if token is None:
                chat_history.append({"role": "assistant", "content": response_text})
                break
            response_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/knowledge')
def knowledge():
    page = request.args.get('page', 1, type=int)
    docs, total = doc_processor.get_documents(page=page, per_page=10)
    total_pages = math.ceil(total / 10)
    return render_template('knowledge.html', 
                         docs=docs, 
                         current_page=page,
                         total_pages=total_pages)

@csrf.exempt
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('knowledge'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('knowledge'))
    try:
        doc_processor.process_file(file)
    except Exception as e:
        return str(e), 400
    return redirect(url_for('knowledge'))

@csrf.exempt
@app.route('/delete/<int:doc_id>', methods=['POST'])
def delete_document(doc_id):
    doc_processor.delete_document(doc_id)
    return redirect(url_for('knowledge'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session_id = session.get('csrf_token')
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return '', 204

if __name__ == '__main__':
    app.run(debug=True) 