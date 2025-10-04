import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from src.financial_statement_analysis.tools.rag_tool import RetrieverTool

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize LLM
retriever_tool = RetrieverTool()

# LLM Setup (unchanged)
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
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
    verbose=True
)

income_statement_agent = Agent(
    role="Financial Data Extraction Specialist",
    goal="Extract specific income statement metrics from financial documents.",
    backstory="You are a seasoned financial analyst with deep expertise in income statement analysis. You can effortlessly navigate complex financial statements to find the exact data points you need.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True
)

balance_sheet_agent = Agent(
    role="Financial Data Extraction Specialist",
    goal="Extract balance sheet metrics from financial documents.",
    backstory="You are a highly experienced financial analyst specializing in balance sheet analysis. Your sharp eye for detail allows you to dissect any balance sheet with precision and accuracy.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True
)

risk_liquidity_agent = Agent(
    role="Financial Risk and Liquidity Analyst",
    goal="Extract key risk and liquidity metrics from the provided financial statement context.",
    backstory="You are a veteran financial risk analyst with a deep understanding of risk management and liquidity metrics. You are adept at identifying and extracting the most critical risk-related data from financial statements.",
    llm=llm,
    tools=[retriever_tool],
    verbose=True
)

# Define tasks
extract_metadata_task = Task(
    description="Extract metadata from the financial statement.",
    agent=metadata_agent,
    expected_output="A JSON object containing the entity name, currency, report year, and previous year."
)

extract_income_statement_task = Task(
    description="Extract income statement metrics from the financial statement.",
    agent=income_statement_agent,
    expected_output="A JSON object containing net interest income, non-interest income, operating expenses, interest expense, and net income for the report year and previous year.",
    context=[extract_metadata_task]
)

extract_balance_sheet_task = Task(
    description="Extract balance sheet metrics from the financial statement.",
    agent=balance_sheet_agent,
    expected_output="A JSON object containing total assets, assets line items, total liabilities, liabilities line items, total equity, and equity line items for the report year and previous year.",
    context=[extract_metadata_task]
)

extract_risk_liquidity_task = Task(
    description="Extract risk and liquidity metrics from the financial statement.",
    agent=risk_liquidity_agent,
    expected_output="A JSON object containing total loans, total deposits, loan loss provisions, non-performing loans, regulatory capital, risk-weighted assets, high-quality liquid assets, and net cash outflows over 30 days for the report year and previous year.",
    context=[extract_metadata_task]
)

# Create crew
crew = Crew(
    agents=[
        metadata_agent,
        income_statement_agent,
        balance_sheet_agent,
        risk_liquidity_agent
    ],
    tasks=[
        extract_metadata_task,
        extract_income_statement_task,
        extract_balance_sheet_task,
        extract_risk_liquidity_task
    ],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)