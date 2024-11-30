import numpy as np
import pandas as pd

from trading_strategy.core.base_strategy import BaseStrategy
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.models.instrument import Instrument

class TrendFollowingStrategy(BaseStrategy):
    def __init__(
        self, 
        instrument: Instrument,
        config: StrategyConfig,
        short_window: int = 20,
        long_window: int = 50
    ):
        super().__init__(instrument, config)
        self.short_window = short_window
        self.long_window = long_window
        self._historical_prices: list = []

    def calculate_position_size(self, market_data: MarketData) -> float:
        """
        Calculate position size based on market volatility and risk parameters
        """
        # Use ATR for volatility-based position sizing
        atr = market_data.atr or self._calculate_atr()
        risk_amount = self.config.initial_capital * self.config.max_position_risk
        return (risk_amount / (atr * 3)) * market_data.close_price

    def _calculate_atr(self, window: int = 14) -> float:
        """
        Calculate Average True Range if not provided in market data
        """
        if len(self._historical_prices) < window:
            return 1.0  # Default value
        
        prices = pd.DataFrame(self._historical_prices[-window:])
        true_ranges = [
            max(
                prices['high'].iloc[i] - prices['low'].iloc[i],
                abs(prices['high'].iloc[i] - prices['close'].iloc[i-1]),
                abs(prices['low'].iloc[i] - prices['close'].iloc[i-1])
            ) for i in range(1, len(prices))
        ]
        
        return np.mean(true_ranges)

    def check_entry_conditions(self, market_data: MarketData) -> bool:
        """
        Trend following entry: Moving average crossover
        """
        self._historical_prices.append({
            'close': market_data.close_price,
            'high': market_data.high_price,
            'low': market_data.low_price
        })
        
        # Keep only recent prices to manage memory
        self._historical_prices = self._historical_prices[-self.long_window:]
        
        if len(self._historical_prices) < self.long_window:
            return False
        
        prices = pd.DataFrame(self._historical_prices)
        
        # Calculate moving averages
        short_ma = prices['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = prices['close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # Entry condition: Short MA crosses above Long MA
        return short_ma > long_ma

    def check_exit_conditions(self, market_data: MarketData) -> bool:
        """
        Trend following exit: Moving average crossover
        """
        if len(self._historical_prices) < self.long_window:
            return False
        
        prices = pd.DataFrame(self._historical_prices)
        
        # Calculate moving averages
        short_ma = prices['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = prices['close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # Exit condition: Short MA crosses below Long MA
        return short_ma < long_ma
