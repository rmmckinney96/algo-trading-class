from functools import lru_cache
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional

from trading_strategy.models.strategy_config import RiskConfig
from trading_strategy.models.market_data import MarketData

class RiskState(BaseModel):
    peak_capital: float
    current_capital: float
    weekly_trades: int = Field(default=0)
    consecutive_losses: int = Field(default=0)
    is_halted: bool = Field(default=False)
    halt_start_time: Optional[datetime] = None
    _drawdown: float = Field(default=0, alias='drawdown')
    
    @property
    def drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        return (self.peak_capital - self.current_capital) / self.peak_capital * 100

class RiskManager:
    def __init__(self, config: RiskConfig, initial_capital: float):
        self.config = config
        self.state = RiskState(
            peak_capital=initial_capital,
            current_capital=initial_capital
        )
    
    @lru_cache(maxsize=1000)
    def _check_waiting_period(self, timestamp: datetime) -> bool:
        """Cached check for waiting period"""
        if not self.state.halt_start_time:
            return True
        return (timestamp - self.state.halt_start_time).total_seconds() / 3600 >= self.config.waiting_period_hours
    
    def can_trade(self, market_data: MarketData) -> bool:
        if self.state.is_halted:
            if self._check_waiting_period(market_data.timestamp):
                self.state.is_halted = False
            else:
                return False
                
        return (
            self._check_drawdown() and 
            self._check_weekly_trades() and
            not self.state.is_halted
        )
    
    def update_state(self, pnl: float, timestamp: datetime):
        self.state.current_capital += pnl
        self.state.peak_capital = max(self.state.peak_capital, self.state.current_capital) 