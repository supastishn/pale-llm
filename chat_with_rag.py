import os  # Ensure os is imported
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Determine the absolute path to the faiss_index directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# Load the vector store
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, oembed, allow_dangerous_deserialization=True)

# Load the LLM
llm = Ollama(
    base_url="http://localhost:11434",
    model="qwen3:0.6b",
    callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0  # Added temperature setting
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

