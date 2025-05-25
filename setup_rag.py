import os
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Changed import
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv # Added import

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

# Create the vector store from all accumulated splits
vectorstore = FAISS.from_documents(all_documents_splits, oembed)

# Save the vector store
vectorstore.save_local("faiss_index")

print("RAG database setup complete.")
