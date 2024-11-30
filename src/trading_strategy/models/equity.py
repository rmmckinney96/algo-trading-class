from datetime import datetime
from pydantic import BaseModel

class EquityPoint(BaseModel):
    """Single point in the equity curve"""
    timestamp: datetime
    equity: float
    unrealized_pnl: float = 0 