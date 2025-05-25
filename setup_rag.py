import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma # Changed import
from dotenv import load_dotenv

# Load environment variables from .env (for GEMINI_API_KEY)
load_dotenv()

# Initialize an empty list to hold all splits from all documents
all_documents_splits = []
data_folder = "data/"

# Loop over all files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        print(f"Processing file: {file_path}")
        # Load the TXT file
        loader = TextLoader(file_path)
        data = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        all_documents_splits.extend(splits)

# Create embeddings using Google Generative AI
# This requires GEMINI_API_KEY to be set in your environment (e.g., .env file)
oembed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

# Create the vector store from all accumulated splits and persist it
CHROMA_DB_PATH = "chroma_db"
if os.path.exists(CHROMA_DB_PATH):
    print(f"Directory {CHROMA_DB_PATH} already exists. Please remove it if you want to re-create the database.")
    # Or, load existing and add:
    # vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=oembed)
    # vectorstore.add_documents(all_documents_splits) # Be careful with duplicates
else:
    vectorstore = Chroma.from_documents(all_documents_splits, oembed, persist_directory=CHROMA_DB_PATH)

print(f"RAG database setup complete. ChromaDB created in '{CHROMA_DB_PATH}'.")
