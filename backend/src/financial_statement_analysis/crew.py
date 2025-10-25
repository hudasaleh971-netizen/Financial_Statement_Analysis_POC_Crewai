# crew.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from src.financial_statement_analysis.tools.vectorstore_load import RetrieverTool
from src.financial_statement_analysis.utils.langfuse_config import init_langfuse

# Import the updated Pydantic models
from src.financial_statement_analysis.utils.pydantic_models import (
    FSUnitAndYears,
    FullBalanceSheet,
    FullIncomeStatement,
    KeyFinancialMetrics,
    KeyIncomeMetrics,
    KeyRiskMetrics
)

# (other imports)

load_dotenv()
# (other initializations like langfuse, output_dir, etc.)

# LLM Setup (ensure your API key is correctly loaded from .env)
# WARNING: Avoid hardcoding API keys in your script.
# The key below is invalid and for structural example only.
# Initialize the LLM using CrewAI's LLM class
llm = LLM(
    model="openai/custom-models/gemma-3-4b-it-Q4_0.gguf",
    base_url="http://127.0.0.1:8080/v1",
    api_key="localdummykey",
    temperature=0,
    lite_llm_extra={
        "drop_params": True,
        "messages": []  # This helps reset message state
    }
)
llm = LLM(
    model="gemini/gemini-2.5-flash",  # or "gemini/gemini-1.5-pro" for more complex tasks
    api_key=os.getenv("GEMINI_API_KEY"),  # Load from .env file
    temperature=0
)
retriever_tool = RetrieverTool()

# --- AGENT DEFINITIONS ---

metadata_agent = Agent(
    role="Financial Metadata Extractor",
    goal="""Extract entity name, currency/unit, report year, previous year, and the fiscal month-end 
          from the introductory sections or headers of the financial document in the file {filename}.""",
    backstory="A specialist in identifying core metadata from financial reports.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True, max_iter=2
)

balance_sheet_agent = Agent(
    role="Balance Sheet Analyst",
    goal="""Find the 'Consolidated Balance Sheet' table in the file {filename}.
          Extract every single line item, its value for the current and previous year.
          **Crucially, classify each item's 'category' using ONLY ONE of these exact strings:**
          - 'Assets line Item'
          - 'Total Asset'
          - 'Liabilities line Item'
          - 'Total Liabilities'
          - 'Equity line Item'
          - 'Total Equity'
          """,
    backstory="A highly detailed accountant specializing in balance sheets who meticulously follows category guidelines and never misses a line item.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
    max_iter=3
)

key_metrics_agent = Agent(
    role="Key Metrics Extractor",
    goal="""From the full balance sheet context, find and extract the values for 'Total Deposits' and 
          'Total Loans' for the current and previous year. For each, also add a 'category' field with the same name as the metric.""",
    backstory="A focused analyst who hones in on 'Total Deposits' and 'Total Loans' from the balance sheet data.",
    llm=llm,
    verbose=True, max_iter=2
)

income_statement_agent = Agent(
    role="Income Statement Analyst",
    goal="""Find the 'Consolidated Income Statement' in the file {filename}. Extract all line items down to 'Net Profit' 
          or 'Profit for the year', along with their values for the current and previous year.""",
    backstory="An expert in profit and loss analysis who systematically extracts the full income statement.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True, max_iter=3
)

income_analysis_agent = Agent(
    role="Income & Provisions Analyst",
    goal="""From the full income statement context, extract and provide values for:
          - 'Net Interest Income', - 'Non-Interest Income', - 'Operating Expenses',
          - 'Interest Expense', - 'Net Income', - 'Loan Loss Provisions'.
          For each, provide a confidence score, a brief explanation, and a 'category' field with the same name as the metric.""",
    backstory="A senior financial analyst who derives key performance indicators from the income statement.",
    llm=llm,
    verbose=True, max_iter=3
)

# --- UPDATED: risk_analysis_agent ---
risk_analysis_agent = Agent(
    role="Financial Risk & Liquidity Analyst",
    goal="""Search the entire document {filename} one by one for the following specific metrics:
          - 'Risk-Weighted Assets'
          - 'Non-Performing Loans'
          - 'Net Cash Outflows (30 days)'
          - 'Regulatory Capital'
          - 'High-Quality Liquid Assets'
          For each, extract the values for the current and previous year, provide a confidence score, and an explanation.""",
    backstory="A specialist in regulatory risk and liquidity reporting who searches for each required metric individually to ensure accuracy.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True, max_iter=4 # Increased iterations for searching multiple individual items
)


# --- TASK DEFINITIONS ---

task_extract_metadata = Task(
    description=
     """
    Extract metadata from the financial statement {filename}:
    1. **Entity name** – Identify the company or organization to which the statement belongs.
    2. **Report year(s)** – Extract the fiscal or calendar year(s) mentioned in the document.
    3. **Report month** – Identify the fiscal month-end (e.g., December, June, etc.) if available.
    4. **Unit** – Extract the measurement unit (e.g., EGP, USD, USD000, EGP000).

    When analyzing the document:
    - Always start with checking the {filename} as it might contain some of the metadata (don't use the tool in the first step)
    - Then use the tool to find the other metadata example searching for entity (company), year, month, and unit (use all amounts in Unit or all figures in Unit to search for unit).
    - If multiple pieces of information (e.g., entity and year) are found in one step, record all of them and skip re-searching those fields.
    - Avoid repeating searches for already identified details.
    - If information is not found proceed to the next
    - If you found more concrete information about already filled metadate field, feel free to edit
    - Ensure the extracted information is specific, accurate, and consistent with the financial statement context.
    """,

    
    expected_output="A JSON object matching the FSUnitAndYears model.",
    agent=metadata_agent,
    output_pydantic=FSUnitAndYears,
    output_file= "src/financial_statement_analysis/output/FSUnitAndYears.md",
)

task_extract_balance_sheet = Task(
    description="""Based on the metadata context, find the 'Consolidated Balance Sheet' table in {filename}.
                 Extract ALL line items and their values for the report_year and prev_year.
                 **Assign the correct category to each item using ONLY these exact strings:**
                 - 'Assets line Item' (for individual asset lines)
                 - 'Total Asset' (for the total assets line)
                 - 'Liabilities line Item' (for individual liability lines)
                 - 'Total Liabilities' (for the total liabilities line)
                 - 'Equity line Item' (for individual equity lines)
                 - 'Total Equity' (for the total equity line)
                 """,
    expected_output="A JSON object matching the FullBalanceSheet model, ensuring all 'category' fields use the exact allowed strings.",
    agent=balance_sheet_agent,
    context=[task_extract_metadata],
    output_file= "src/financial_statement_analysis/output/FullBalanceSheet.md",
    output_pydantic=FullBalanceSheet
)

task_extract_key_metrics = Task(
    description=(
        "From the provided balance sheet context, identify all relevant items that represent "
        "either deposits or loans. "
        "For each item, extract:\n"
        "  - metric_name: the label exactly as written in the balance sheet\n"
        "  - category: choose **only one** of ['Total Deposits', 'Total Loans'] based on meaning\n"
        "  - current_year_value: the numeric value for the most recent year\n"
        "  - previous_year_value: the numeric value for the previous year\n\n"
        "Examples:\n"
        "  'Customers’ deposits' → category: 'Total Deposits'\n"
        "  'Loans and advances to customers (net)' → category: 'Total Loans'\n\n"
        "Ensure the final output strictly follows the JSON schema of KeyFinancialMetrics."
    ),
    expected_output="A JSON object matching the KeyFinancialMetrics model.",
    agent=key_metrics_agent,
    context=[task_extract_balance_sheet],
    output_pydantic=KeyFinancialMetrics,
    output_file= "src/financial_statement_analysis/output/KeyFinancialMetrics.md",
)

task_extract_income_statement = Task(
    description="""Using the metadata context, find the 'Consolidated Income Statement' table in {filename}.
                   Extract all line items down to 'Net Profit'.
                   For each item, extract:\n"
                    - metric_name: the label exactly as written in the Income Statement\n
                    - category: choose **only one** of ['income item', 'income derived item'] based on metric\n
                    - current_year_value: the numeric value for the most recent year\n
                    - previous_year_value: the numeric value for the previous year\n\n""",

    expected_output="A JSON object matching the FullIncomeStatement model.",
    agent=income_statement_agent,
    context=[task_extract_metadata],
    output_pydantic=FullIncomeStatement,
    output_file= "src/financial_statement_analysis/output/FullIncomeStatement.md",
)

task_analyze_income_metrics = Task(
    description="""
        From the full income statement context, extract the following key income metrics:
        - "Net Interest Income"
        - "Non-Interest Income"
        - "Operating Expenses"
        - "Interest Expense"
        - "Net Income"
        - "Loan Loss Provisions"

        **Requirements:**
        1. Each output item must contain:
            - 'metric_name' → exactly one of the names listed above.
            - 'category' → the same as 'metric_name' (exact string match).
            - current_year_value: the numeric value for the most recent year\n
            - previous_year_value: the numeric value for the previous yea
            - 'confidence' → a float between 0 and 1.
            - 'explanation' → a short text explaining how the metric was found or derived.

        2. If a metric is not explicitly found:
            - Attempt to derive it from available values (e.g., summing or subtracting related items).
            - Clearly describe the performed calculation in the 'explanation' field.
            - Lower the 'confidence' score (e.g., 0.6 or less) if the value is estimated or uncertain.""",
    expected_output="A JSON object matching the KeyIncomeMetrics model.",
    agent=income_analysis_agent,
    context=[task_extract_income_statement],
    output_pydantic=KeyIncomeMetrics,
    output_file= "src/financial_statement_analysis/output/KeyIncomeMetrics.md",
)

# --- UPDATED: task_analyze_risk_metrics ---
task_analyze_risk_metrics = Task(
    description="""
                Analyze the financial document {filename} to extract or initialize the following risk and liquidity metrics:

                - "Risk-Weighted Assets"
                - "Non-Performing Loans"
                - "Net Cash Outflows (30 days)"
                - "Regulatory Capital"
                - "High-Quality Liquid Assets"

                **Search Strategy:**
                1. For each metric, perform a targeted search.
                2. If the initial query fails, refine it once only
                (e.g., for "Non-Performing Loans", try "Impaired Loans").
                3. If the metric still cannot be found, include it in the output with
                `current_year_value` and `previous_year_value` left as `null`.
                In that case:
                    - Set `confidence` = 0.0
                    - Set `explanation` = "Metric not found in the document."

                **Use Metadata Context:**
                - Use the context from `task_extract_metadata` to determine:
                    - `report_year` and `prev_year` for aligning values.
                    - `unit` (e.g., EGP, USD, millions) for consistent interpretation.          
                 """,
    expected_output="""A JSON object matching the KeyRiskMetrics model. The 'metric_name' for each item in the list
                     must be one of: 'Risk-Weighted Assets', 'Non-Performing Loans', 'Net Cash Outflows (30 days)',
                     'Regulatory Capital', or 'High-Quality Liquid Assets'.""",
    agent=risk_analysis_agent,
    context=[task_extract_metadata],
    output_pydantic=KeyRiskMetrics
)


# --- CREW DEFINITION ---

crew = Crew(
    agents=[
        metadata_agent,
        balance_sheet_agent,
        key_metrics_agent,
        income_statement_agent,
        income_analysis_agent,
        risk_analysis_agent
    ],
    tasks=[
        task_extract_metadata,
        task_extract_balance_sheet,
        task_extract_key_metrics,
        task_extract_income_statement,
        task_analyze_income_metrics,
        task_analyze_risk_metrics
    ],
    process=Process.sequential,
    verbose=True
)

# (rest of the script for running the crew)

if __name__ == "__main__":
    import pprint
    import json

    # Initialize Langfuse, instrumentation for tracing
    langfuse = init_langfuse()  # Add this call; returns the langfuse client

    filename = "egypt_publish_f_s_dec_2024_separate_en_q4.pdf"

    # The inputs dictionary will be used to fill the {filename} placeholder
    inputs = {'filename': filename}

    # Wrap the crew kickoff in a Langfuse span for tracing
    with langfuse.start_as_current_span(name="financial-analysis-crew"):
        result = crew.kickoff(inputs=inputs)

    # Flush Langfuse events (important for short-lived scripts)
    langfuse.flush()

    print("\n" + "="*50)
    print("## Complete Financial Analysis Results ##")
    print("="*50 + "\n")
    
    # The final aggregated result
    if result.pydantic:
        print("✓ Complete Financial Analysis (Aggregated):")
        pprint.pprint(result.pydantic.dict(), indent=2)
        
        # Save to JSON file
        with open('financial_analysis_complete.json', 'w') as f:
            json.dump(result.pydantic.dict(), f, indent=2)
        print("\n✓ Results saved to 'financial_analysis_complete.json'")
    else:
        print("⚠ Pydantic output not available. Raw output:")
        print(result.raw)
    
    print("\n" + "="*50)
    
    # Optional: Print individual task outputs for debugging
    print("\n## Individual Task Outputs ##")
    for i, task_output in enumerate(result.tasks_output):
        print(f"\n--- Task {i+1} ---")
        if hasattr(task_output, 'pydantic') and task_output.pydantic:
            print("Pydantic Output:")
            pprint.pprint(task_output.pydantic.dict(), indent=2)
        if hasattr(task_output, 'raw'):
            print("\nRaw Output (first 500 chars):")
            print(str(task_output.raw)[:500] + "...")
