# Financial Statement Analysis RAG Agent Crew

This project implements a sophisticated multi-agent system using CrewAI to perform detailed analysis of financial statement documents. It leverages a Retrieval-Augmented Generation (RAG) pipeline to extract, analyze, and structure key financial metrics from PDF documents.

## Features

- **Multi-Agent System**: Utilizes a crew of specialized AI agents (e.g., Metadata Specialist, Income Statement Analyst) to perform distinct analysis tasks.
- **RAG Pipeline**: Implements a full RAG pipeline for information retrieval from financial documents using a Qdrant vector store.
- **Structured Data Output**: Enforces structured, validated output for all analysis tasks using Pydantic models.
- **Declarative Crew Definition**: Uses CrewAI's modern `@CrewBase` decorator pattern for a clean and scalable project structure.
- **Configuration-Based**: Agents and tasks are defined in simple YAML files for easy modification and expansion.
- **Advanced Document Processing**: Employs the `docling` library for robust PDF parsing and the `nomic-ai/nomic-embed-text-v1.5` model for high-quality embeddings.

## Technology Stack

- **Orchestration**: [CrewAI](https://github.com/crewAIInc/crewAI)
- **LLM Interaction**: [LangChain](https://www.langchain.com/)
- **Vector Store**: [Qdrant](https://qdrant.tech/)
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` via Hugging Face
- **Document Processing**: `docling`
- **Data Validation**: Pydantic
- **LLM**: Google Gemini

## Project Structure

```
backend/src/financial_statement_analysis/
├── config/             # YAML definitions for agents and tasks
│   ├── agents.yaml
│   └── tasks.yaml
├── tools/              # Custom tools for the crew
│   └── vectorstore_load.py
├── utils/              # Utility scripts and models
│   ├── document_chunker.py
│   ├── document_processor.py
│   └── pydantic_models.py
├── crew_improved.py    # Main application: Defines and runs the CrewAI crew
└── ingest.py           # Standalone script for the data ingestion pipeline
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8+
- A Google API key with the Gemini API enabled.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Set up Environment

It is highly recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 4. Install Dependencies

The required packages can be found in `backend/pyproject.toml`. You can install them using `pip`.

```bash
pip install crewai langchain-google-genai langchain-huggingface transformers "pydantic>=2.0" qdrant-client docling torch
```

### 5. Configure Environment Variables

Create a `.env` file inside the `backend/` directory:

```
backend/.env
```

Add your Google API key to this file:

```
GOOGLE_API_KEY="your_google_api_key"
```

## Usage Workflow

Running the analysis is a two-step process: data ingestion and crew execution.

### Step 1: Ingest Your Document

First, you need to process your financial PDF and load it into the Qdrant vector store. Run the `ingest.py` script from the project root directory, providing the path to your document.

```bash
python backend/src/financial_statement_analysis/ingest.py "path/to/your/financial_statement.pdf"
```

This command will:
1.  Process the PDF.
2.  Chunk the content and generate AI-powered descriptions for tables.
3.  Create embeddings using the `nomic-embed-text-v1.5` model.
4.  Save the results into a local Qdrant database located at `backend/qdrant_db/`.

### Step 2: Run the Analysis Crew

Once the data is ingested, you can run the analysis crew. Execute the `crew_improved.py` script, passing the **filename** of the document you just ingested.

```bash
python backend/src/financial_statement_analysis/crew_improved.py "financial_statement.pdf"
```

The crew will then kick off, with each agent performing its specialized task on the document data, returning a final structured JSON object with the full analysis.

## Configuration

You can easily modify the behavior of the crew without changing the Python code:

-   **Agents**: Edit `backend/src/financial_statement_analysis/config/agents.yaml` to change the roles, goals, or backstories of your AI agents.
-   **Tasks**: Edit `backend/src/financial_statement_analysis/config/tasks.yaml` to change the descriptions or expected outputs for each task.
