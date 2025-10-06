from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.financial_statement_analysis.tools.vectorstore_load import RetrieverTool
from src.financial_statement_analysis.utils.pydantic_models import (
    FSCurrencyAndYears,
    IncomeStatementMetrics,
    BalanceSheetMetrics,
    RiskAndLiquidityMetrics
)
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
retriever_tool = RetrieverTool()


# Using the @CrewBase decorator to define the crew.
@CrewBase
class FinancialAnalysisCrew():
    """A crew designed to analyze financial statements from a given file."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        self.retriever_tool = RetrieverTool()
        # LLM Setup (unchanged)
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0,
            api_key=GOOGLE_API_KEY
        )

    # Agent definitions using the @agent decorator
    @agent
    def metadata_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['metadata_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm
        )

    @agent
    def income_statement_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['income_statement_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm

        )

    @agent
    def balance_sheet_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['balance_sheet_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm

        )

    @agent
    def risk_liquidity_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_liquidity_agent'],
            tools=[self.retriever_tool],
            verbose=True,
            llm=self.llm

        )

    # Task definitions using the @task decorator
    @task
    def extract_metadata(self) -> Task:
        return Task(
            config=self.tasks_config['extract_metadata'],
            agent=self.metadata_agent(),
            output_pydantic=FSCurrencyAndYears
        )

    @task
    def extract_income_statement(self) -> Task:
        return Task(
            config=self.tasks_config['extract_income_statement'],
            agent=self.income_statement_agent(),
            output_pydantic=IncomeStatementMetrics
        )

    @task
    def extract_balance_sheet(self) -> Task:
        return Task(
            config=self.tasks_config['extract_balance_sheet'],
            agent=self.balance_sheet_agent(),
            output_pydantic=BalanceSheetMetrics
        )

    @task
    def extract_risk_liquidity(self) -> Task:
        return Task(
            config=self.tasks_config['extract_risk_liquidity'],
            agent=self.risk_liquidity_agent(),
            output_pydantic=RiskAndLiquidityMetrics
        )

    # Crew definition using the @crew decorator
    @crew
    def crew(self) -> Crew:
        """Creates the financial analysis crew"""
        return Crew(
            agents=self.agents,  # These are created by the @agent decorator
            tasks=self.tasks,    # These are created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

# Main execution block to run the crew
if __name__ == '__main__':
    import argparse
    import pprint
    filename="HSBC-11.pdf"


    # The inputs dictionary will be used to fill the {filename} placeholder
    inputs = {'filename': filename}

    # Instantiate the crew
    financial_crew = FinancialAnalysisCrew()

    # Kick off the crew with the provided inputs
    result = financial_crew.crew().kickoff(inputs)

    print("\n" + "="*50)
    print("## Financial Analysis Result ##")
    print("="*50 + "\n")
    pprint.pprint(result)
    print("\n" + "="*50)