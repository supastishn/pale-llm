from flask import Flask, render_template, request, Response, stream_with_context, jsonify # Added jsonify
import sys
import os
import json
import threading
import queue

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI # Changed import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv # Added import
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables from .env (for GEMINI_API_KEY)
load_dotenv()

# Determine the absolute path to the chroma_db directory (relative to the project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db") # Changed path

# --- Custom Callback Handler for Web Streaming ---
class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token is not None: # Ensure token is not None
            self.q.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        # This signals the end of tokens from LLM, but RetrievalQA might add sources later.
        # The main thread will handle putting None after chain completion.
        pass

    def on_chain_end(self, outputs, **kwargs) -> None:
        # This could be a place to get source_documents if they are in outputs
        pass


# --- Initialize Langchain components (globally or per request as needed) ---
try:
    # Use Google Generative AI Embeddings, consistent with setup_rag.py
    oembed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB directory not found at {CHROMA_DB_PATH}. Please run setup_rag.py first.")
        vectorstore = None
    else:
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=oembed)
except Exception as e:
    print(f"Error loading ChromaDB or Google Generative AI Embeddings: {e}")
    # Fallback or error state if these fail. For now, app might not work.
    vectorstore = None

prompt_template_str = """The answer may not be in the context. If the user's prompt is unrelated to the context, you do not need to use context.
Use the following pieces of context to answer the question at the end only if the user's prompt is related.. If the prompt is not related, don't use the context. Simply respond normally.
Example prompts that are unrelated:
Hello!
What is the weather like today?
How are you?
Write me some code to do X
Example prompts that are related:
What is the Nakba?
How many people were refugees because of the Nakba?
What is Camp David?
What is 1972 war in egypt?
Context: {context}
User's Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['GET']) # Changed from POST to GET
def chat_endpoint():
    user_message = request.args.get('message') # Changed from request.json.get to request.args.get for GET requests
    temperature = request.args.get('temperature', '0.7')  # Get temperature from query params, default to 0.7
    
    if not user_message:
        return jsonify({'error': 'No message provided in query parameters'}), 400
    
    # Validate and convert temperature
    try:
        temperature = float(temperature)
        temperature = max(0.0, min(2.0, temperature))  # Clamp between 0 and 2
    except (ValueError, TypeError):
        temperature = 0.7  # Default fallback
    
    if not vectorstore:
        return jsonify({'error': 'Vectorstore not initialized. Check server logs.'}), 500

    token_q = queue.Queue()
    q_callback = QueueCallbackHandler(token_q)

    # Configure LLM to match chat_with_rag.py with dynamic temperature
    llm_for_stream = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        model_name="gpt-4.1",  # Matches chat_with_rag.py
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        streaming=True,
        callbacks=[q_callback],
        temperature=temperature
    )
    current_qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_stream, chain_type="stuff", retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
    )

    final_result_holder = {} # To store sources from the thread

    def run_chain_in_thread():
        try:
            result = current_qa_chain.invoke({"query": user_message})
            final_result_holder['sources'] = [doc.metadata.get('source', 'Unknown source') for doc in result.get("source_documents", [])]
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            token_q.put(json.dumps({'error': f"Error in RAG chain: {str(e)}"})) # Send error as a JSON string token
        finally:
            token_q.put(None) # Signal end of processing from this thread

    thread = threading.Thread(target=run_chain_in_thread)
    thread.start()

    def generate_sse_stream():
        try:
            while True:
                item = token_q.get()
                if item is None: # End of stream signal from thread
                    break
                
                # Check if item is an error JSON string from the queue
                try:
                    error_payload = json.loads(item)
                    if 'error' in error_payload:
                        yield f"data: {json.dumps(error_payload)}\n\n"
                        continue # Skip further processing for this item
                except (json.JSONDecodeError, TypeError):
                    # Not an error JSON string, treat as a regular token
                    pass

                yield f"data: {json.dumps({'token': item})}\n\n"
            
            thread.join() # Ensure thread is finished before sending sources

            if 'sources' in final_result_holder:
                yield f"data: {json.dumps({'sources': final_result_holder['sources']})}\n\n"
            
            yield f"data: {json.dumps({'event': 'eos'})}\n\n" # End Of Stream event for client
        except Exception as e:
            print(f"Error in SSE stream generator: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    return Response(stream_with_context(generate_sse_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=8005)
