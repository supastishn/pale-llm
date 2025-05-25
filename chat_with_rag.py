import os  # Ensure os is imported
from langchain_openai import ChatOpenAI # Replaced Ollama with ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Changed import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env
load_dotenv()

# Determine the absolute path to the faiss_index directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# Load the vector store using Google Generative AI Embeddings
# This requires GEMINI_API_KEY to be set in your environment (e.g., .env file)
oembed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, oembed, allow_dangerous_deserialization=True)

# Load the LLM using ChatOpenAI for an external OpenAI-compatible API
# IMPORTANT: Replace "your-custom-base-url-here" with your actual API base URL.
# IMPORTANT: Replace "your-model-name-here" with the model name your API expects.
# Ensure the OPENAI_API_KEY environment variable is set if your API requires authentication.
llm = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"), # Defaults to OpenAI's standard API base if OPENAI_API_BASE env var is not set
    model_name="gpt-4.1",          # e.g., "gpt-4.1", "gpt-3.5-turbo", or other model served by your endpoint
    openai_api_key=os.getenv("OPENAI_API_KEY", ""), # Must be set for OpenAI models
    streaming=True,
    callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0
)

# Create the RAG chain
retriever = vectorstore.as_retriever()

prompt_template = """The answer may not be in the context. If the user's prompt is unrelated to the context, you do not need to use context.

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

def chat(query):
    print("Answer: ", end="", flush=True)
    response = qa_chain.invoke({"query": query})
    print()
    print("Sources:", [doc.metadata.get('source', 'Unknown source') for doc in response["source_documents"]])

if __name__ == "__main__":
    print("Starting chat with RAG. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        chat(user_input)

# This takes a while to load

