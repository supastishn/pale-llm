# THE DEVELOPER OF THIS PROGRAM IS AGAINST THE GENOCIDE IN GAZA.
[Learn More](https://www.amnesty.org/en/wp-content/uploads/2024/12/MDE1586682024ENGLISH.pdf)

# Local RAG Chat with Ollama

This project demonstrates how to set up a local Retrieval Augmented Generation (RAG) system using Ollama, LangChain, and FAISS. It allows you to chat with a local Large Language Model (LLM) that can reference a custom knowledge base (in this case, `data/text.txt`).

## Features

*   Uses Ollama to run LLMs locally.
*   Embeds documents using the `nomic-embed-text` model.
*   Stores embeddings in a FAISS vector database.
*   Allows chatting with the `qwen3:0.6b` model, augmented with information from the local text file.
*   Easily adaptable to use different text files or PDFs.

## Project Structure

```
.
├── chat_with_rag.py    # Script to chat with the RAG system
├── data/
│   └── texts.txt        # Sample text file for the RAG knowledge base
├── faiss_index/        # Directory where the FAISS vector store is saved (created by setup_rag.py)
├── requirements.txt    # Python dependencies
├── setup_rag.py        # Script to process the data and create the RAG database
└── README.md           # This file
```

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama:** Ensure Ollama is installed and running. You can download it from [ollama.com](https://ollama.com/).
3.  **Ollama Models:** Pull the required models:
    ```bash
    ollama pull nomic-embed-text
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
    Run the setup script. This will process the document in `data/text.txt`, generate embeddings, and save them into the `faiss_index` directory.
    ```bash
    python setup_rag.py
    ```
    This step might take a few minutes depending on the size of your document and your system's performance.

## Usage

Once the setup is complete, you can start chatting with the RAG-powered LLM:

```bash
python chat_with_rag.py
```

The script will prompt you for input. Type your questions and press Enter. To exit the chat, type `exit`.

## Customization

*   **Different LLMs:** You can change the chat model (`qwen3:0.6b`) or the embedding model (`nomic-embed-text`) in `chat_with_rag.py` and `setup_rag.py` respectively. Make sure the models are available in your Ollama instance.
*   **Different Data Source:**
    *   To use a different `.txt` file, update the `file_path` in `setup_rag.py` (line `loader = TextLoader("data/your_file.txt")`).
    *   To use a PDF file:
        1.  Change `from langchain_community.document_loaders import TextLoader` to `from langchain_community.document_loaders import PyPDFLoader` in `setup_rag.py`.
        2.  Change `loader = TextLoader("data/text.txt")` to `loader = PyPDFLoader("data/your_pdf_file.pdf")` in `setup_rag.py`.
        3.  Ensure `pypdf` is in your `requirements.txt`.
*   **Chunking Strategy:** You can adjust the `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` in `setup_rag.py` to fine-tune how the document is split.

## Files

*   `setup_rag.py`: Loads the specified document (`data/text.txt` by default), splits it into chunks, generates embeddings using `nomic-embed-text` via Ollama, and stores these embeddings in a FAISS vector database saved locally in the `faiss_index` directory.
*   `chat_with_rag.py`: Loads the pre-built FAISS vector database and the `qwen3:0.6b` chat model via Ollama. It sets up a retrieval chain that fetches relevant document chunks based on your query and provides them as context to the LLM to generate an answer.
*   `requirements.txt`: Lists all necessary Python packages for the project.
*   `data/text.txt`: A sample text file used as the knowledge base for the RAG system.
*   `faiss_index/`: This directory is created by `setup_rag.py` and contains the FAISS vector store files (`index.faiss` and `index.pkl`). **Do not edit these files manually.** If you change your data source, delete this directory and re-run `setup_rag.py`.

## Chat Guide

I used the `qwen3:0.6b` model as it is light weight enough to use in Github Codespaces.

However, higher parameter models (1.7b, 4b, 8b) are recommended, as the 0.6b model often gets things wrong.

To enable thinking, type:
/think (your query)

To disable thinking, type:
/no_think (your query)
