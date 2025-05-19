from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the vector store
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", oembed, allow_dangerous_deserialization=True)

# Load the LLM
llm = Ollama(base_url="http://localhost:11434", model="qwen3:0.6b")

# Create the RAG chain
retriever = vectorstore.as_retriever()

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

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

def chat(query):
    response = qa_chain.invoke({"query": query})
    print("Answer:", response["result"])
    print("Sources:", [doc.metadata.get('source', 'Unknown source') for doc in response["source_documents"]])

if __name__ == "__main__":
    print("Starting chat with RAG. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        chat(user_input)
