from typing import List
from trading_strategy.core.strategy import TradingStrategy
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.position import Position, PositionSide
from trading_strategy.indicators.technical import calculate_indicators_vectorized
from trading_strategy.indicators.registry import IndicatorRegistry
import numpy as np

# Register a custom indicator
@IndicatorRegistry.register('momentum')
def calculate_momentum(prices: np.ndarray, period: int) -> np.ndarray:
    """Custom momentum indicator"""
    momentum = np.zeros_like(prices)
    momentum[period:] = prices[period:] - prices[:-period]
    return momentum

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.price_history: List[MarketData] = []
        self.indicator_configs = {
            'sma_20': {'period': 20},
            'sma_50': {'period': 50},
            'atr_14': {'period': 14},
            'momentum_10': {'period': 10}  # Easy to add new indicators
        }
    
    def calculate_indicators(self, market_data: MarketData) -> MarketData:
        # Store price history for indicator calculations
        self.price_history.append(market_data)
        
        # Calculate all indicators at once using vectorized function
        # Get maximum period from indicator configs
        max_period = max(
            config['period'] for config in self.indicator_configs.values()
        )
        
        if len(self.price_history) >= max_period:
            indicators = calculate_indicators_vectorized(
                self.price_history, 
                self.indicator_configs
            )[-1]  # Get latest values
            market_data.indicators.update(indicators)
        
        # Update equity curve with unrealized PnL if we have an open position
        if self.position:
            unrealized_pnl = (
                (market_data.close - self.position.entry_price) 
                if self.position.side == PositionSide.LONG 
                else (self.position.entry_price - market_data.close)
            ) * self.position.size
            self.update_equity(market_data.timestamp, unrealized_pnl)
        
        return market_data
    
    def check_entry_conditions(self, market_data: MarketData) -> bool:
        # Basic trend following entry: price crosses above SMA
        if not market_data.indicators.get("sma_20"):
            return False
            
        return market_data.close > market_data.indicators["sma_20"]
    
    def check_exit_conditions(self, market_data: MarketData) -> bool:
        # Exit when price crosses below SMA
        if not market_data.indicators.get("sma_20"):
            return False
            
        return market_data.close < market_data.indicators["sma_20"]
    
    def calculate_position_size(self, market_data: MarketData) -> float:
        """Calculate position size based on risk parameters and current market data"""
        # Get ATR for volatility-based position sizing
        atr = market_data.indicators.get("atr")
        
        # Calculate risk amount in account currency
        risk_amount = self.current_equity * self.config.risk_config.max_position_risk
        
        # Calculate position size based on ATR as stop distance
        if atr and atr > 0:
            position_size = risk_amount / atr
        else:
            position_size = 0
        
        return position_size
