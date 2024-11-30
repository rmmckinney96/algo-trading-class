from abc import ABC, abstractmethod
from typing import List, Dict, Any

from trading_strategy.models.market_data import MarketData
from trading_strategy.models.indicators import TechnicalIndicators
from trading_strategy.core.position_manager import PositionManager
from trading_strategy.models.strategy_config import StrategyConfig

class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.position_manager = PositionManager(config)
        
    @abstractmethod
    def calculate_indicators(self, market_data: MarketData) -> TechnicalIndicators:
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def check_entry_conditions(
        self, 
        market_data: MarketData, 
        indicators: TechnicalIndicators
    ) -> bool:
        """Define entry logic"""
        pass
    
    @abstractmethod
    def check_exit_conditions(
        self, 
        market_data: MarketData, 
        indicators: TechnicalIndicators
    ) -> bool:
        """Define exit logic"""
        pass
