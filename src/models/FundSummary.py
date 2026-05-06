from pydantic import BaseModel
from typing import Literal

class FundSummary(BaseModel):
    name: str
    strategy: str
    risk_level: Literal["low", "medium", "high"]
    minimum_investment_usd: int
    summary: str