
# GEMINI.md

## Project Overview

This project is a proof-of-concept for a financial statement analysis system using a Retrieval-Augmented Generation (RAG) architecture. It leverages the CrewAI framework to create a team of AI agents that can analyze financial documents and answer user queries.

The system works by first processing PDF financial statements, converting them into a searchable format, and then using a RAG pipeline to retrieve relevant information to answer user questions.

**Core Technologies:**

*   **CrewAI:** A framework for orchestrating autonomous AI agents.
*   **LangChain:** A library for building applications with large language models (LLMs).
*   **Docling:** A library for document conversion, chunking, and OCR.
*   **FAISS:** A library for efficient similarity search and clustering of dense vectors, used here as a vector store.
*   **Hugging Face Transformers:** Used for generating embeddings.
*   **Google Gemini:** The large language model used for generation tasks.

## Building and Running

### 1. Prerequisites

*   Python 3.x
*   An environment with the required Python packages installed. While a `requirements.txt` or `pyproject.toml` was not found, the following packages are used in the code:
    *   `crewai`
    *   `langchain`
    *   `langchain-google-genai`
    *   `langchain-community`
    *   `langchain-huggingface`
    *   `faiss-cpu` or `faiss-gpu`
    *   `docling`
    *   `torch`
    *   `transformers`
    *   `python-dotenv`
    *   `loguru`
    *   `sentence-transformers`
    *   `easyocr`

### 2. Setup

1.  **Install Dependencies:**
    ```bash
    # It is recommended to create a virtual environment first.
    # pip install -r requirements.txt # (A requirements.txt file would be ideal here)
    pip install crewai langchain langchain-google-genai langchain-community langchain-huggingface faiss-cpu docling torch transformers python-dotenv loguru sentence-transformers easyocr
    ```

2.  **Set up Environment Variables:**
    Create a `.env` file in the root of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key"
    ```

### 3. Running the System

The project is divided into two main parts: data preparation and the agentic RAG system.

**Part 1: Data Preparation**

The `FS_Preperation.py` script is used to process a PDF financial statement and create a FAISS vector store.

To run it, you need to modify the `pdf_file` variable in the `if __name__ == "__main__":` block of the script to point to the PDF file you want to process.

```python
# In FS_Preperation.py
if __name__ == "__main__":
    pdf_file = "path/to/your/financial_statement.pdf" # <-- CHANGE THIS
    
    try:
        # Process and store PDF
        vector_store = load_and_store_pdf(pdf_file)
        print("✓ Successfully processed and stored PDF in FAISS")
        
        # ...
            
    except Exception as e:
        print(f"❌ Error: {e}")
```

Then, run the script from your terminal:

```bash
python FS_Preperation.py
```

This will create a `faiss_index` directory in the root of the project.

**Part 2: Running the Agentic RAG System**

The `test.py` script demonstrates how to use the CrewAI agent to query the financial document.

To run it, you can modify the `test_query` variable in the `if __name__ == "__main__":` block of the script to ask a question about the document.

```python
# In test.py
if __name__ == "__main__":
    # Test the system
    test_query = "What were the total revenues?" # <-- CHANGE THIS
    try:
        response = chat_with_agent(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
```

Then, run the script from your terminal:

```bash
python test.py
```

The script will then print the agent's response to your query.

## Development Conventions

*   **Logging:** The project uses the `loguru` library for logging. Logs are stored in the `backend/logs` directory.
*   **Modular Design:** The project is divided into separate files for data preparation (`FS_Preperation.py`), the agentic RAG system (`FS_Agentic_RAG.py`), and testing (`test.py`).
*   **CrewAI:** The `test.py` script demonstrates the use of CrewAI to define agents and tasks.
*   **Tooling:** The project defines a custom `RetrieverTool` for CrewAI that retrieves documents from the FAISS vector store.
