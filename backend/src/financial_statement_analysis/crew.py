import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from src.financial_statement_analysis.tools.vectorstore_load import RetrieverTool
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import date
from src.financial_statement_analysis.utils.pydantic_models import (
    FSUnitAndYears,
    IncomeStatementMetrics,
    BalanceSheetMetrics,
    RiskAndLiquidityMetrics,
    CompleteFinancialAnalysis
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Setup
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

# Initialize the retriever tool
retriever_tool = RetrieverTool()

# Define agents
metadata_agent = Agent(
    role="Financial Data Extraction Specialist",
    goal="Extract key metadata from the cover page, headers, or introductory sections of a financial document.",
    backstory="You are a meticulous and detail-oriented financial analyst with a knack for quickly identifying and extracting critical information from financial reports.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
    max_iter=1
)

income_statement_agent = Agent(
    role="Financial Data Extraction Specialist",
    goal="Extract specific income statement metrics from financial documents.",
    backstory="You are a seasoned financial analyst with deep expertise in income statement analysis. You can effortlessly navigate complex financial statements to find the exact data points you need.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
    max_iter=2
)

balance_sheet_agent = Agent(
    role="Financial Data Extraction Specialist",
    goal="Extract balance sheet metrics from financial documents.",
    backstory="You are a highly experienced financial analyst specializing in balance sheet analysis. Your sharp eye for detail allows you to dissect any balance sheet with precision and accuracy.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
    max_iter=2
)

risk_liquidity_agent = Agent(
    role="Financial Risk and Liquidity Analyst",
    goal="Extract key risk and liquidity metrics from the provided financial statement context.",
    backstory="You are a veteran financial risk analyst with a deep understanding of risk management and liquidity metrics. You are adept at identifying and extracting the most critical risk-related data from financial statements.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True,
    max_iter=1
)

aggregator_agent = Agent(
    role="Senior Financial Data Aggregator and Validator",
    goal="Review all previous task outputs (both structured Pydantic data and raw text) and combine them into a single, complete, and accurate financial analysis. If any Pydantic fields are empty or missing, extract the data from the raw text outputs.",
    backstory="You are a meticulous financial analyst with expertise in data validation and aggregation. You have a keen eye for detail and can parse through both structured and unstructured data. Your job is to ensure that no valuable information is lost - if the Pydantic output is blank but the raw text contains the data, you extract it and structure it properly. You cross-reference all outputs to ensure consistency and completeness.",
    llm=llm,
    verbose=True,
    max_iter=2
)

# Define tasks
extract_metadata_task = Task(
    description="Extract metadata from the financial statement in the file {filename}. Parse values in parentheses like (100) as negative floats like -100 where applicable. Use None for any missing values.",
    expected_output="A JSON object matching the FSUnitAndYears model with keys: entity_name (string), unit (string), report_year (integer), prev_year (integer).",
    agent=metadata_agent,
    # output_pydantic=FSUnitAndYears,
    output_format="json"
)

extract_income_statement_task = Task(
    description="Extract income statement metrics from the financial statement in the file {filename}. Parse values in parentheses like (100) as negative floats like -100. Use None for any missing values.",
    expected_output="A JSON object matching the IncomeStatementMetrics model with keys like net_interest_income (dict of year to optional float), and similar for non_interest_income, operating_expenses, interest_expense, net_income.",
    agent=income_statement_agent,
    # output_pydantic=IncomeStatementMetrics,
    context=[extract_metadata_task],
    output_format="json"
)

extract_balance_sheet_task = Task(
    description="Extract balance sheet metrics from the financial statement in the file {filename}. Parse values in parentheses like (100) as negative floats like -100. Use None for any missing values.",
    expected_output="A JSON object matching the BalanceSheetMetrics model with keys like total_assets (dict of year to optional float), assets_line_items (dict of str to dict of year to optional float), and similar for liabilities and equity.",
    agent=balance_sheet_agent,
    # output_pydantic=BalanceSheetMetrics,
    context=[extract_metadata_task],
    output_format="json"
)

extract_risk_liquidity_task = Task(
    description="Extract risk and liquidity metrics from the financial statement in the file {filename}. Parse values in parentheses like (100) as negative floats like -100. Use None for any missing values.",
    expected_output="A JSON object matching the RiskAndLiquidityMetrics model with keys like total_loans (dict of year to optional float), and similar for other risk and liquidity metrics.",
    agent=risk_liquidity_agent,
    # output_pydantic=RiskAndLiquidityMetrics,
    context=[extract_metadata_task],
    output_format="json"
)

aggregate_task = Task(
    description="""Review ALL previous task outputs from the metadata, income statement, balance sheet, 
    and risk/liquidity extraction tasks. 
    
    IMPORTANT: Each previous task has TWO outputs:
    1. A structured Pydantic object (which may be incomplete or empty)
    2. A raw text output (which contains the full details)
    
    Your job is to:
    - Examine both the structured and raw text outputs from each task
    - If the Pydantic output is missing data but the raw text has it, extract it
    - Validate that all extracted values are correct and consistent
    - Combine everything into a single CompleteFinancialAnalysis object
    - Ensure no data is lost in the aggregation process""",
    expected_output="""A complete CompleteFinancialAnalysis Pydantic object containing:
    - metadata: FSUnitAndYears with entity_name, unit, report_year, prev_year
    - income_statement: IncomeStatementMetrics with all available metrics
    - balance_sheet: BalanceSheetMetrics with all available metrics and line items
    - risk_and_liquidity: RiskAndLiquidityMetrics with all available metrics
    
    All fields should be populated with accurate data extracted from previous tasks.""",
    agent=aggregator_agent,
    output_pydantic=CompleteFinancialAnalysis,
    context=[
        extract_metadata_task,
        extract_income_statement_task,
        extract_balance_sheet_task,
        extract_risk_liquidity_task
    ],
    output_format="json"
)

# Create crew
crew = Crew(
    agents=[
        metadata_agent,
        income_statement_agent,
        balance_sheet_agent,
        risk_liquidity_agent,
        aggregator_agent
    ],
    tasks=[
        extract_metadata_task,
        extract_income_statement_task,
        extract_balance_sheet_task,
        extract_risk_liquidity_task,
        aggregate_task
    ],
    process=Process.sequential,
    verbose=True,
    max_rpm=30
)

if __name__ == "__main__":
    import pprint
    import json

    filename = "HSBC-11.pdf"

    # The inputs dictionary will be used to fill the {filename} placeholder
    inputs = {'filename': filename}

    # Kick off the crew with the provided inputs
    result = crew.kickoff(inputs=inputs)
    
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