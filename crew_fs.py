import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI  # Replace with ChatGoogleGenerativeAI or Groq if needed
from pydantic_models import (
    FSCurrencyAndYears,
    IncomeStatementMetrics,
    BalanceSheetMetrics,
    RiskAndLiquidityMetrics
)
from prompts import (
    METADATA_PROMPT,
    INCOME_STATEMENT_PROMPT,
    BALANCE_SHEET_PROMPT,
    RISK_LIQUIDITY_PROMPT
)
import logging
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM Setup ---
# Using OpenAI as example; replace with Google GenAI or Groq as per your original code

import os
from crewai import LLM

# Read your API key from the environment variable
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Use Gemini 2.5 Pro Experimental model
llm = LLM(
    model='gemini/gemini-1.5-flash',
    api_key=gemini_api_key,
    temperature=0.0  # Lower temperature for more consistent results.
)
# Alternative: Groq (uncomment if using)
# from langchain_groq import ChatGroq
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, api_key=GROQ_API_KEY)

# --- Load FAISS Vector Store ---
def load_existing_faiss(faiss_index_path: str = "./faiss_index") -> FAISS:
    """Load an existing FAISS index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

try:
    vector_store = load_existing_faiss()
    logger.info("Successfully loaded FAISS vector store.")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")
    logger.error("Please ensure you have created the index by running the data preparation script first.")
    exit()

# --- Define RAG Tool ---
def rag_tool_fn(query: str) -> str:
    """Retrieve relevant context from the vector store based on the query."""
    docs = vector_store.similarity_search(query, k=5)  # Retrieve top 5 relevant chunks
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Relevant Financial Context:\n{context}"

rag_tool = Tool(
    name="RAG Retriever",
    description="Use this tool to retrieve relevant sections from the financial document based on a query. Input is a search query string.",
    func=rag_tool_fn
)

# --- Define Agents ---
metadata_agent = Agent(
    role="Financial Metadata Extractor",
    goal="Extract key metadata like entity name, currency, report year, and previous year from the financial document.",
    backstory="You are a specialist in identifying core metadata from financial reports using retrieval tools.",
    tools=[rag_tool],
    llm=llm,
    verbose=True
)

income_agent = Agent(
    role="Income Statement Analyst",
    goal="Extract income statement metrics for specified years and currency from the financial document.",
    backstory="You are an expert in parsing income statements and extracting precise financial metrics using retrieval tools.",
    tools=[rag_tool],
    llm=llm,
    verbose=True
)

balance_sheet_agent = Agent(
    role="Balance Sheet Analyst",
    goal="Extract balance sheet metrics, including all line items, for specified years and currency.",
    backstory="You are specialized in analyzing balance sheets and categorizing assets, liabilities, and equity using retrieval tools.",
    tools=[rag_tool],
    llm=llm,
    verbose=True
)

risk_liquidity_agent = Agent(
    role="Risk and Liquidity Analyst",
    goal="Extract risk and liquidity metrics for specified years and currency from the financial document.",
    backstory="You are an expert in identifying risk-related metrics like loans, deposits, and capital requirements using retrieval tools.",
    tools=[rag_tool],
    llm=llm,
    verbose=True
)

# --- Define Tasks (Sequential Execution) ---
# Task 1: Extract Metadata
metadata_task = Task(
    description=str(METADATA_PROMPT).format(
        pydantic_schema=FSCurrencyAndYears.model_json_schema(),
        context_str="{Use the RAG Retriever tool to fetch relevant context for metadata extraction.}"
    ),
    expected_output="A valid JSON object matching the FSCurrencyAndYears Pydantic schema. Do not include any additional text.",
    agent=metadata_agent
)

# Task 2: Extract Income Statement (depends on metadata for years/currency)
income_task = Task(
    description="First, parse the metadata output from the previous task to get report_year, prev_year, and currency. "
                "Then, use these to extract income statement metrics. "
                + str(INCOME_STATEMENT_PROMPT).format(
                    report_year="{report_year from metadata}",
                    prev_year="{prev_year from metadata}",
                    currency="{currency from metadata}",
                    pydantic_schema=IncomeStatementMetrics.model_json_schema(),
                    context_str="{Use the RAG Retriever tool to fetch relevant income statement sections.}"
                ),
    expected_output="A valid JSON object matching the IncomeStatementMetrics Pydantic schema. Do not include any additional text.",
    agent=income_agent,
    context=[metadata_task]  # Use output from metadata_task
)

# Task 3: Extract Balance Sheet (depends on metadata)
balance_sheet_task = Task(
    description="First, parse the metadata output from the previous task to get report_year, prev_year, and currency. "
                "Then, use these to extract balance sheet metrics. "
                + str(BALANCE_SHEET_PROMPT).format(
                    report_year="{report_year from metadata}",
                    prev_year="{prev_year from metadata}",
                    currency="{currency from metadata}",
                    pydantic_schema=BalanceSheetMetrics.model_json_schema(),
                    context_str="{Use the RAG Retriever tool to fetch relevant balance sheet sections.}"
                ),
    expected_output="A valid JSON object matching the BalanceSheetMetrics Pydantic schema. Do not include any additional text.",
    agent=balance_sheet_agent,
    context=[metadata_task]  # Use output from metadata_task
)

# Task 4: Extract Risk & Liquidity (depends on metadata)
risk_liquidity_task = Task(
    description="First, parse the metadata output from the previous task to get report_year, prev_year, and currency. "
                "Then, use these to extract risk and liquidity metrics. "
                + str(RISK_LIQUIDITY_PROMPT).format(
                    report_year="{report_year from metadata}",
                    prev_year="{prev_year from metadata}",
                    currency="{currency from metadata}",
                    pydantic_schema=RiskAndLiquidityMetrics.model_json_schema(),
                    context_str="{Use the RAG Retriever tool to fetch relevant risk and liquidity sections.}"
                ),
    expected_output="A valid JSON object matching the RiskAndLiquidityMetrics Pydantic schema. Do not include any additional text.",
    agent=risk_liquidity_agent,
    context=[metadata_task]  # Use output from metadata_task
)

# --- Define Crew (Sequential Process) ---
extraction_crew = Crew(
    agents=[metadata_agent, income_agent, balance_sheet_agent, risk_liquidity_agent],
    tasks=[metadata_task, income_task, balance_sheet_task, risk_liquidity_task],
    verbose=True  # For detailed logging
)

# --- Run the Crew ---
if __name__ == "__main__":
    try:
        result = extraction_crew.kickoff()  # Runs tasks in sequence
        print("Final Result:", result)
        
        # Optionally, parse and print individual outputs
        metadata_output = json.loads(metadata_task.output.raw_output)
        print("\nMetadata:", metadata_output)
        
        income_output = json.loads(income_task.output.raw_output)
        print("\nIncome Statement:", income_output)
        
        balance_output = json.loads(balance_sheet_task.output.raw_output)
        print("\nBalance Sheet:", balance_output)
        
        risk_output = json.loads(risk_liquidity_task.output.raw_output)
        print("\nRisk & Liquidity:", risk_output)
    except Exception as e:
        logger.error(f"Error running crew: {e}")