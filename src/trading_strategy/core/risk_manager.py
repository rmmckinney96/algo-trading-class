from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional

from trading_strategy.models.strategy_config import RiskConfig
from trading_strategy.models.market_data import MarketData

class RiskState(BaseModel):
    peak_capital: float
    current_capital: float
    weekly_trades: int
    consecutive_losses: int
    is_halted: bool
    halt_start_time: Optional[datetime] = None

class RiskManager:
    def __init__(self, config: RiskConfig, initial_capital: float):
        self.config = config
        self.state = RiskState(
            peak_capital=initial_capital,
            current_capital=initial_capital,
            weekly_trades=0,
            consecutive_losses=0,
            is_halted=False
        )
    
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