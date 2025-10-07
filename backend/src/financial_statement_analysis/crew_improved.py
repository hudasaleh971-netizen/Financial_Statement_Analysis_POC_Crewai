import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.financial_statement_analysis.tools.vectorstore_load import RetrieverTool
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import date

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define individual Pydantic Models
class FSUnitAndYears(BaseModel):
    """
    Pydantic model for capturing metadata from a financial statement.
    """
    entity_name: str = Field(..., description="The name of the company or entity, e.g., 'HSBC Group'")
    unit: str = Field(..., description="The unit of measurement for financial values, e.g., 'US$000', 'AED millions'")
    report_year: int = Field(..., ge=1900, le=2100, description="The main reporting fiscal year")
    prev_year: int = Field(..., ge=1900, le=2100, description="The previous fiscal year for comparison")

class IncomeStatementMetrics(BaseModel):
    """
    Pydantic model for income statement metrics.
    Parse values in parentheses like (100) as negative floats like -100.
    """
    net_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Net income from interest-earning assets. Use None if not found for a year.")
    non_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Income from non-core banking activities. Use None if not found for a year.")
    operating_expenses: Dict[int, Optional[float]] = Field(default_factory=dict, description="Expenses incurred during normal business operations. Use None if not found for a year.")
    interest_expense: Dict[int, Optional[float]] = Field(default_factory=dict, description="Cost of borrowed funds. Use None if not found for a year.")
    net_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="The bottom line, representing total profit. Use None if not found for a year.")

class BalanceSheetMetrics(BaseModel):
    """
    Pydantic model for balance sheet metrics.
    Parse values in parentheses like (100) as negative floats like -100.
    """
    total_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total value of all assets owned by the company. Use None if not found for a year.")
    assets_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of asset components, e.g., {'Cash': {2023: 1000.0, 2022: None}}.")
    total_liabilities: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total obligations of the company. Use None if not found for a year.")
    liabilities_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of liability components.")
    total_equity: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total shareholders' equity. Use None if not found for a year.")
    equity_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of equity components.")

class RiskAndLiquidityMetrics(BaseModel):
    """
    Pydantic model for risk and liquidity metrics.
    Parse values in parentheses like (100) as negative floats like -100.
    """
    total_loans: Dict[int, Optional[float]] = Field(default_factory=dict, description="Also known as 'Loans and advances to customers'. Use None if not found for a year.")
    total_deposits: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total amount of money held in deposit accounts. Use None if not found for a year.")
    loan_loss_provisions: Dict[int, Optional[float]] = Field(default_factory=dict, description="An expense set aside for uncollected loans. Use None if not found for a year.")
    non_performing_loans: Dict[int, Optional[float]] = Field(default_factory=dict, description="Loans in default or close to default. Use None if not found for a year.")
    regulatory_capital: Dict[int, Optional[float]] = Field(default_factory=dict, description="The amount of capital a bank is required to hold. Use None if not found for a year.")
    risk_weighted_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="A bank's assets, weighted by risk. Use None if not found for a year.")
    high_quality_liquid_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="Assets that can be easily converted to cash. Use None if not found for a year.")
    net_cash_outflows_over_30_days: Dict[int, Optional[float]] = Field(default_factory=dict, description="Projected net cash outflows over a 30-day period. Use None if not found for a year.")

# Combined Pydantic Model
class CompleteFinancialAnalysis(BaseModel):
    """
    Combined Pydantic model that aggregates all financial statement analysis results.
    This model should be populated by reviewing ALL previous task outputs (both structured Pydantic data and raw text).
    If any Pydantic fields are missing or empty, extract the data from the raw text output.
    """
    metadata: FSUnitAndYears = Field(..., description="Entity metadata including name, unit, and reporting years")
    income_statement: IncomeStatementMetrics = Field(..., description="Income statement metrics")
    balance_sheet: BalanceSheetMetrics = Field(..., description="Balance sheet metrics")
    risk_and_liquidity: RiskAndLiquidityMetrics = Field(..., description="Risk and liquidity metrics")

# Using the @CrewBase decorator to define the crew.
@CrewBase
class FinancialAnalysisCrew():
    """A crew designed to analyze financial statements from a given file."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        self.retriever_tool = RetrieverTool()
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0,
            api_key=GEMINI_API_KEY
        )

    # Agent definitions using the @agent decorator
    @agent
    def metadata_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['metadata_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm,
            max_iter=1
        )

    @agent
    def income_statement_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['income_statement_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm,
            max_iter=2
        )

    @agent
    def balance_sheet_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['balance_sheet_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm,
            max_iter=2
        )

    @agent
    def risk_liquidity_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_liquidity_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm,
            max_iter=1
        )

    @agent
    def aggregator_agent(self) -> Agent:
        """
        Agent that reviews all previous task outputs and combines them into a complete analysis.
        This agent will parse raw text outputs if Pydantic parsing failed for any task.
        """
        return Agent(
            config=self.agents_config['aggregator_agent'],
            verbose=True,
            llm=self.llm,
            max_iter=2  # Allow re-work if needed
        )

    # Task definitions using the @task decorator
    @task
    def extract_metadata(self) -> Task:
        return Task(
            config=self.tasks_config['extract_metadata'],
            agent=self.metadata_agent(),
            output_pydantic=FSUnitAndYears
        )

    @task
    def extract_income_statement(self) -> Task:
        return Task(
            config=self.tasks_config['extract_income_statement'],
            agent=self.income_statement_agent(),
            output_pydantic=IncomeStatementMetrics,
            context=[self.extract_metadata()]
        )

    @task
    def extract_balance_sheet(self) -> Task:
        return Task(
            config=self.tasks_config['extract_balance_sheet'],
            agent=self.balance_sheet_agent(),
            output_pydantic=BalanceSheetMetrics,
            context=[self.extract_metadata()]
        )

    @task
    def extract_risk_liquidity(self) -> Task:
        return Task(
            config=self.tasks_config['extract_risk_liquidity'],
            agent=self.risk_liquidity_agent(),
            output_pydantic=RiskAndLiquidityMetrics,
            context=[self.extract_metadata()]
        )

    @task
    def aggregate_all_results(self) -> Task:
        """
        Final aggregation task that reviews ALL previous outputs (structured and raw text)
        and produces a complete, validated financial analysis.
        """
        return Task(
            config=self.tasks_config['aggregate_all_results'],
            agent=self.aggregator_agent(),
            output_pydantic=CompleteFinancialAnalysis,
            context=[
                self.extract_metadata(),
                self.extract_income_statement(),
                self.extract_balance_sheet(),
                self.extract_risk_liquidity()
            ]
        )

    # Crew definition using the @crew decorator
    @crew
    def crew(self) -> Crew:
        """Creates the financial analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_rpm=30
        )

# Main execution block to run the crew
if __name__ == '__main__':
    import pprint
    import json
    
    filename = "HSBC-11.pdf"

    # The inputs dictionary will be used to fill the {filename} placeholder
    inputs = {'filename': filename}

    # Instantiate the crew
    financial_crew = FinancialAnalysisCrew()

    # Kick off the crew with the provided inputs
    result = financial_crew.crew().kickoff(inputs)

    print("\n" + "="*50)
    print("## Complete Financial Analysis Results ##")
    print("="*50 + "\n")

    # The final result from the aggregator task
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