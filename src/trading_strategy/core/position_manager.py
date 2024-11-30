from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.models.market_data import MarketData

class Position(BaseModel):
    entry_price: float
    size: float
    entry_time: datetime
    highest_price: float = Field(default=0.0)
    trailing_stop: float = Field(default=0.0)
    
    def update(self, current_price: float) -> None:
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.trailing_stop = current_price * (1 - self.trailing_stop_pct)

class PositionManager:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.current_position: Optional[Position] = None
        
    def calculate_position_size(self, price: float, atr: float) -> float:
        risk_amount = self.config.initial_capital * self.config.risk_config.max_position_risk
        return (risk_amount / (atr * 3.0)) * price * self.config.leverage 