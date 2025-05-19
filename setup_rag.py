from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load the TXT file
loader = TextLoader("data/text.txt")
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)

# Create embeddings
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# Create the vector store
vectorstore = FAISS.from_documents(all_splits, oembed)

# Save the vector store
vectorstore.save_local("faiss_index")

print("RAG database setup complete.")
