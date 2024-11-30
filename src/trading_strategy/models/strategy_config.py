from pydantic import BaseModel, Field, validator
from typing import Optional

class RiskConfig(BaseModel):
    max_position_risk: float = Field(0.02, description="Maximum risk per trade")
    max_drawdown: float = Field(0.10, description="Maximum allowed drawdown")
    waiting_period_hours: int = Field(72, description="Waiting period after drawdown")
    max_weekly_trades: int = Field(5, description="Maximum trades per week")
    trailing_stop_pct: Optional[float] = Field(None, description="Trailing stop percentage")

class StrategyConfig(BaseModel):
    initial_capital: float = Field(..., description="Initial trading capital")
    risk_config: RiskConfig = Field(default_factory=RiskConfig)
    symbol: str = Field(..., description="Trading symbol")
    indicators: list[str] = Field(default_factory=list, description="Required indicators")
    
    @validator('initial_capital')
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError("Initial capital must be positive")
        return v
