# THE DEVELOPER OF THIS PROGRAM IS AGAINST THE GENOCIDE IN GAZA.
[Learn More](https://www.amnesty.org/en/wp-content/uploads/2024/12/MDE1586682024ENGLISH.pdf)

# Local RAG Chat with Ollama

This project demonstrates how to set up a local Retrieval Augmented Generation (RAG) system using a compatible LLM (like a local one via Ollama or a cloud one via an API endpoint), LangChain, Google Gemini Embeddings, and ChromaDB. It allows you to chat with an LLM that can reference a custom knowledge base.

## Features

*   Uses a configurable LLM for chat (defaulting to an OpenAI-compatible API endpoint).
*   Embeds documents using Google Gemini Embeddings (`models/embedding-001`).
*   Stores embeddings in a ChromaDB vector database.
*   Allows chatting with the configured LLM, augmented with information from local text files.
*   Easily adaptable to use different text files or PDFs.

## Project Structure

```
.
├── chat_with_rag.py    # Script to chat with the RAG system
├── data/
│   └── texts.txt        # Sample text file for the RAG knowledge base
├── chroma_db/          # Directory where the ChromaDB vector store is saved (created by setup_rag.py)
├── requirements.txt    # Python dependencies
├── setup_rag.py        # Script to process the data and create the RAG database
└── README.md           # This file
```

## Prerequisites

1.  **Python 3.8+**
2.  **Google API Key:** You need a Google API key with the Generative Language API enabled. Set this key as an environment variable `GEMINI_API_KEY`. You can create a `.env` file in the project root:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
3.  **Ollama (Optional for local LLM):** If you intend to use a local LLM via Ollama for the chat model (instead of the default OpenAI-compatible API), ensure Ollama is installed and running. You can download it from [ollama.com](https://ollama.com/).
4.  **Ollama Models (Optional):** If using Ollama for chat, pull your desired chat model, e.g.:
    ```bash
    ollama pull qwen3:0.6b
    ```

## Setup

1.  **Clone the repository (if applicable) or download the files.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    *   The default setup uses `data/*.txt`. You can replace these files or modify `setup_rag.py` to point to a different `.txt` file or a PDF.
    *   If using a PDF, you'll need to adjust `setup_rag.py` to use `PyPDFLoader` instead of `TextLoader`.

5.  **Build the RAG database:**
    Run the setup script. This will process the documents in `data/*.txt`, generate embeddings using Google Gemini, and save them into the `chroma_db` directory.
    ```bash
    python setup_rag.py
    ```
    If the `chroma_db` directory already exists, the script will print a message and exit to avoid overwriting. Remove the directory if you want to rebuild the database.
    This step might take a few minutes depending on the size of your documents and your system's performance. Remember to have your `GEMINI_API_KEY` set.

## Usage

Once the setup is complete, you can start chatting with the RAG-powered LLM:

```bash
python chat_with_rag.py
```

The script will prompt you for input. Type your questions and press Enter. To exit the chat, type `exit`.

## Customization

*   **Different LLMs:** You can change the chat model configuration in `chat_with_rag.py` (e.g., `model_name`, `openai_api_base`). The embedding model is now Google Gemini.
*   **Different Data Source:**
    *   The script `setup_rag.py` processes all `.txt` files in the `data/` directory.
    *   To use a PDF file:
        1.  Change `from langchain_community.document_loaders import TextLoader` to `from langchain_community.document_loaders import PyPDFLoader` in `setup_rag.py`.
        2.  Change `loader = TextLoader("data/text.txt")` to `loader = PyPDFLoader("data/your_pdf_file.pdf")` in `setup_rag.py`.
        3.  Ensure `pypdf` is in your `requirements.txt`.
*   **Chunking Strategy:** You can adjust the `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` in `setup_rag.py` to fine-tune how the document is split.

## Files

*   `setup_rag.py`: Loads all `*.txt` documents from the `data/` directory, splits them into chunks, generates embeddings using Google Gemini (`models/embedding-001`), and stores these embeddings in a ChromaDB vector database saved locally in the `chroma_db` directory. Requires `GEMINI_API_KEY`.
*   `chat_with_rag.py`: Loads the pre-built ChromaDB vector database (using Google Gemini embeddings) and the configured chat model (defaulting to an OpenAI-compatible API endpoint). It sets up a retrieval chain that fetches relevant document chunks based on your query and provides them as context to the LLM to generate an answer. Requires `GEMINI_API_KEY` for embeddings and potentially other keys/configs for the chat LLM.
*   `requirements.txt`: Lists all necessary Python packages for the project, including `langchain-google-genai` and `chromadb`.
*   `data/`: Directory containing text files used as the knowledge base for the RAG system.
*   `chroma_db/`: This directory is created by `setup_rag.py` and contains the ChromaDB vector store files (e.g., SQLite database and Parquet files). **Do not edit these files manually.** If you change your data source, delete this directory and re-run `setup_rag.py`.

## Chat Guide

I used the `qwen3:0.6b` model as it is light weight enough to use in Github Codespaces.

However, higher parameter models (1.7b, 4b, 8b) are recommended, as the 0.6b model often gets things wrong.

To enable thinking, type:
/think (your query)

To disable thinking, type:
/no_think (your query)
