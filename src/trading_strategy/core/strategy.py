from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime

from trading_strategy.models.market_data import MarketData
from trading_strategy.models.position import Position, PositionSide
from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.models.equity import EquityPoint

class TradingStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position: Optional[Position] = None
        self.trades: List[Position] = []
        self.equity_curve: List[EquityPoint] = [
            EquityPoint(
                timestamp=datetime.now(),
                equity=config.initial_capital,
                unrealized_pnl=0
            )
        ]
    
    @property
    def current_equity(self) -> float:
        """Get the current equity value from the equity curve"""
        return self.equity_curve[-1].equity
    
    def update_equity(self, timestamp: datetime, unrealized_pnl: float = 0):
        """Update equity curve with new equity value"""
        realized_pnl = sum(t.pnl for t in self.trades) if self.trades else 0
        current_equity = self.config.initial_capital + realized_pnl + unrealized_pnl
        
        self.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                equity=current_equity,
                unrealized_pnl=unrealized_pnl
            )
        )
    
    @abstractmethod
    def calculate_indicators(self, market_data: MarketData) -> MarketData:
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def check_entry_conditions(self, market_data: MarketData) -> bool:
        """Define entry logic"""
        pass
    
    @abstractmethod
    def check_exit_conditions(self, market_data: MarketData) -> bool:
        """Define exit logic"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, market_data: MarketData) -> float:
        """Calculate position size based on risk parameters"""
        pass
    
    def open_position(self, market_data: MarketData, size: float, side: PositionSide = PositionSide.LONG):
        """Open a new position"""
        if self.position is not None:
            return
            
        self.position = Position(
            entry_price=market_data.close,
            size=size,
            side=side,
            entry_time=market_data.timestamp
        )
        
        # Update equity curve with zero unrealized PnL for new position
        self.update_equity(market_data.timestamp, 0)
    
    def close_position(self, market_data: MarketData) -> Optional[Position]:
        """Close the current position and update equity"""
        if self.position is None:
            return None
            
        # Calculate PnL
        exit_price = market_data.close
        if self.position.side == PositionSide.LONG:
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size
        
        # Update position with exit details
        self.position.exit_price = exit_price
        self.position.exit_time = market_data.timestamp
        self.position.pnl = pnl
        
        # Store trade and clear current position
        closed_position = self.position
        self.trades.append(closed_position)
        self.position = None
        
        # Update equity curve with zero unrealized PnL after closing
        self.update_equity(market_data.timestamp, 0)
        
        return closed_position