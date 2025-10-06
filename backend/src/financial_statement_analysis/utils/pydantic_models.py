from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import date

class FSCurrencyAndYears(BaseModel):
    """
    Pydantic model for capturing metadata from a financial statement.
    """
    entity_name: str = Field(..., description="The name of the company or entity, e.g., 'HSBC Group'")
    currency: str = Field(..., description="Primary currency used in the financial statement, e.g., 'USD', 'AED'")
    report_year: int = Field(..., ge=1900, le=2100, description="The main reporting fiscal year")
    prev_year: int = Field(..., ge=1900, le=2100, description="The previous fiscal year for comparison")

class IncomeStatementMetrics(BaseModel):
    """
    Pydantic model for income statement metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    """
    net_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Net income from interest-earning assets. Value is None if not found for a year.")
    non_interest_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="Income from non-core banking activities. Value is None if not found for a year.")
    operating_expenses: Dict[int, Optional[float]] = Field(default_factory=dict, description="Expenses incurred during normal business operations. Value is None if not found for a year.")
    interest_expense: Dict[int, Optional[float]] = Field(default_factory=dict, description="Cost of borrowed funds. Value is None if not found for a year.")
    net_income: Dict[int, Optional[float]] = Field(default_factory=dict, description="The bottom line, representing total profit. Value is None if not found for a year.")

class BalanceSheetMetrics(BaseModel):
    """
    Pydantic model for balance sheet metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    """
    total_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total value of all assets owned by the company. Value is None if not found for a year.")
    assets_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of asset components, e.g., {'Cash': {2023: 1000.0, 2022: None}}.")
    total_liabilities: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total obligations of the company. Value is None if not found for a year.")
    liabilities_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of liability components.")
    total_equity: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total shareholders' equity. Value is None if not found for a year.")
    equity_line_items: Dict[str, Dict[int, Optional[float]]] = Field(default_factory=dict, description="Detailed dictionary of equity components.")
class RiskAndLiquidityMetrics(BaseModel):
    """
    Pydantic model for risk and liquidity metrics. All values are dictionaries
    mapping the year to the financial value, with None if a value is not found for that year.
    """
    total_loans: Dict[int, Optional[float]] = Field(default_factory=dict, description="Also known as 'Loans and advances to customers'. Value is None if not found for a year.")
    total_deposits: Dict[int, Optional[float]] = Field(default_factory=dict, description="Total amount of money held in deposit accounts. Value is None if not found for a year.")
    loan_loss_provisions: Dict[int, Optional[float]] = Field(default_factory=dict, description="An expense set aside for uncollected loans. Value is None if not found for a year.")
    non_performing_loans: Dict[int, Optional[float]] = Field(default_factory=dict, description="Loans in default or close to default. Value is None if not found for a year.")
    regulatory_capital: Dict[int, Optional[float]] = Field(default_factory=dict, description="The amount of capital a bank is required to hold. Value is None if not found for a year.")
    risk_weighted_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="A bank's assets, weighted by risk. Value is None if not found for a year.")
    high_quality_liquid_assets: Dict[int, Optional[float]] = Field(default_factory=dict, description="Assets that can be easily converted to cash. Value is None if not found for a year.")
    net_cash_outflows_30d: Dict[int, Optional[float]] = Field(default_factory=dict, description="Projected net cash outflows over a 30-day period. Value is None if not found for a year.")

class FSExtractionOutput(BaseModel):
    """
    A comprehensive model that aggregates all extracted financial data points.
    """
    metadata: Optional[FSCurrencyAndYears] = None
    income: Optional[IncomeStatementMetrics] = None
    balance_sheet: Optional[BalanceSheetMetrics] = None
    risk_liquidity: Optional[RiskAndLiquidityMetrics] = None
    extraction_timestamp: date = Field(default_factory=date.today)
    confidence_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Per-metric confidence scores, if available.")
