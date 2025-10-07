from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import date


# Define Updated Pydantic Models (renamed currency to unit)
class FSUnitAndYears(BaseModel):
    """
    Pydantic model for capturing metadata from a financial statement.
    This model structures the expected output for the metadata extraction task.
    """
    entity_name: str = Field(..., description="The name of the company or entity, e.g., 'HSBC Group'")
    unit: str = Field(..., description="The unit of measurement for financial values, e.g., 'US$000', 'AED millions'")
    report_year: int = Field(..., ge=1900, le=2100, description="The main reporting fiscal year")
    prev_year: int = Field(..., ge=1900, le=2100, description="The previous fiscal year for comparison")

class IncomeStatementMetrics(BaseModel):
    """
    Pydantic model for income statement metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    This model structures the expected output for the income statement extraction task.
    Parse values in parentheses like (100) as negative floats like -100.
    """
    net_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Net income from interest-earning assets. Use None if not found for a year.")
    non_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Income from non-core banking activities. Use None if not found for a year.")
    operating_expenses: Dict[int, Optional[float]] = Field(default_factory=dict, description="Expenses incurred during normal business operations. Use None if not found for a year.")
    interest_expense: Dict[int, Optional[float]] = Field(default_factory=dict, description="Cost of borrowed funds. Use None if not found for a year.")
    net_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="The bottom line, representing total profit. Use None if not found for a year.")

class BalanceSheetMetrics(BaseModel):
    """
    Pydantic model for balance sheet metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    This model structures the expected output for the balance sheet extraction task.
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
    Pydantic model for risk and liquidity metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    This model structures the expected output for the risk and liquidity extraction task.
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
    """
    metadata: FSUnitAndYears = Field(..., description="Entity metadata including name, unit, and reporting years")
    income_statement: IncomeStatementMetrics = Field(..., description="Income statement metrics")
    balance_sheet: BalanceSheetMetrics = Field(..., description="Balance sheet metrics")
    risk_and_liquidity: RiskAndLiquidityMetrics = Field(..., description="Risk and liquidity metrics")

