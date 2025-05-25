from appwrite.client import Client
from appwrite.services.users import Users
from appwrite.exception import AppwriteException
import os
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

# This Appwrite function will be executed every time your function is triggered
def main(context):
    # Handle CORS for frontend requests
    if context.req.method == "OPTIONS":
        return context.res.empty().headers({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        })

    # The req object contains the request data
    if context.req.path == "/ping":
        return context.res.text("Pong").headers({
            "Access-Control-Allow-Origin": "*"
        })
    
    if context.req.path == "/chat" and context.req.method == "POST":
        try:
            # Parse request body
            body = json.loads(context.req.body) if context.req.body else {}
            message = body.get("message", "")
            
            if not message:
                return context.res.json({"error": "No message provided"}, 400).headers({
                    "Access-Control-Allow-Origin": "*"
                })
            
            # Initialize embeddings (you'll need GEMINI_API_KEY in environment)
            oembed = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=os.environ.get("GEMINI_API_KEY")
            )
            
            # For this demo, we'll create a simple vectorstore with minimal data
            # In production, you'd load your pre-built FAISS index
            from langchain.schema import Document
            docs = [
                Document(page_content="This is a demo RAG system for Palestine education.", metadata={"source": "demo"}),
                Document(page_content="The Nakba refers to the mass displacement of Palestinians in 1948.", metadata={"source": "demo"})
            ]
            vectorstore = FAISS.from_documents(docs, oembed)
            
            # Initialize LLM (you'll need OPENAI_API_KEY in environment)
            callback_handler = StreamingCallbackHandler()
            llm = ChatOpenAI(
                openai_api_base=os.environ.get("OPENAI_API_BASE"),
                model_name="gpt-3.5-turbo",
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0,
                callbacks=CallbackManager([callback_handler])
            )
            
            # Create RAG chain
            retriever = vectorstore.as_retriever()
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}
Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            # Get response
            response = qa_chain.invoke({"query": message})
            
            return context.res.json({
                "response": response["result"],
                "sources": [doc.metadata.get('source', 'Unknown') for doc in response.get("source_documents", [])]
            }).headers({
                "Access-Control-Allow-Origin": "*"
            })
            
        except Exception as e:
            context.error(f"Chat error: {str(e)}")
            return context.res.json({"error": str(e)}, 500).headers({
                "Access-Control-Allow-Origin": "*"
            })

    return context.res.json(
        {
            "motto": "Build like a team of hundreds_",
            "learn": "https://appwrite.io/docs",
            "connect": "https://appwrite.io/discord",
            "getInspired": "https://builtwith.appwrite.io",
        }
    ).headers({
        "Access-Control-Allow-Origin": "*"
    })
