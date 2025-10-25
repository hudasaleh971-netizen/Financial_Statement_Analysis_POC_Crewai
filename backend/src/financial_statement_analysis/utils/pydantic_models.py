# src/financial_statement_analysis/utils/pydantic_models.py
from pydantic import BaseModel, Field, conint, confloat
from typing import List, Optional, Literal
from enum import Enum

# --- Enums for Categorization (No changes) ---

class BalanceSheetCategory(str, Enum):
    ASSETS_LINE_ITEM = "Assets line Item"
    TOTAL_ASSETS = "Total Asset"
    LIABILITIES_LINE_ITEM = "Liabilities line Item"
    TOTAL_LIABILITIES = "Total Liabilities"
    EQUITY_LINE_ITEM = "Equity line Item"
    TOTAL_EQUITY = "Total Equity"

class IncomeStatementCategory(str, Enum):
    INCOME_ITEM = "income item"
    INCOME_DERIVED_ITEM = "income derived item"

# --- Literal Types for Name Validation ---

KeyFinancialMetricName = Literal["Total Deposits", "Total Loans"]

KeyIncomeMetricName = Literal[
    "Net Interest Income",
    "Non-Interest Income",
    "Operating Expenses",
    "Interest Expense",
    "Net Income",
    "Loan Loss Provisions"
]

# --- NEW: Literal for Key Risk Metrics ---
KeyRiskMetricName = Literal[
    "Risk-Weighted Assets",
    "Non-Performing Loans",
    "Net Cash Outflows (30 days)",
    "Regulatory Capital",
    "High-Quality Liquid Assets"
]

# --- Core Data Structures ---

class FinancialMetric(BaseModel):
    metric_name: str = Field(..., description="The name of the financial metric.")
    current_year_value: Optional[float] = Field(None, description="The metric's value for the current reporting year.")
    previous_year_value: Optional[float] = Field(None, description="The metric's value for the previous reporting year.")

class AnalyzedMetric(FinancialMetric):
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="The confidence score (0.0 to 1.0).")
    explanation: str = Field(..., description="A brief explanation of how the value was found or derived.")

# --- Item Models with Constraints ---

class BalanceSheetItem(FinancialMetric):
    category: BalanceSheetCategory

class IncomeStatementItem(FinancialMetric):
    category: IncomeStatementCategory

class KeyFinancialMetricItem(FinancialMetric):
    # --- ADDED: Category field for filtering ---
    category: KeyFinancialMetricName

class KeyIncomeMetricItem(AnalyzedMetric):
    metric_name: KeyIncomeMetricName
    # --- ADDED: Category field for filtering (Note: "Net profit" changed to "Net Income" to match Literal) ---
    category: KeyIncomeMetricName

# --- NEW: Item model for Key Risk Metrics ---
class KeyRiskMetricItem(AnalyzedMetric):
    metric_name: KeyRiskMetricName


# --- Main Agent Output Models ---

class FSUnitAndYears(BaseModel):
    entity_name: str
    unit: str
    report_year: int
    prev_year: int
    fiscal_month_end: int

class FullBalanceSheet(BaseModel):
    metrics: List[BalanceSheetItem] = Field(default_factory=list)

class FullIncomeStatement(BaseModel):
    metrics: List[IncomeStatementItem] = Field(default_factory=list)

class KeyFinancialMetrics(BaseModel):
    metrics: List[KeyFinancialMetricItem]

class KeyIncomeMetrics(BaseModel):
    metrics: List[KeyIncomeMetricItem]

# --- UPDATED: KeyRiskMetrics now uses the new constrained item ---
class KeyRiskMetrics(BaseModel):
    metrics: List[KeyRiskMetricItem]