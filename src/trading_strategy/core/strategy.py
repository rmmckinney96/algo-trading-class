from abc import ABC, abstractmethod
from typing import Optional, Dict

from trading_strategy.models.market_data import MarketData
from trading_strategy.models.position import Position, Trade
from trading_strategy.models.strategy_config import StrategyConfig

class TradingStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[Dict] = []
        
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