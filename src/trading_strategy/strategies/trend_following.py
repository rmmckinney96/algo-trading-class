from trading_strategy.core.strategy import TradingStrategy
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.position import Position, PositionSide

class TrendFollowingStrategy(TradingStrategy):
    def calculate_indicators(self, market_data: MarketData) -> MarketData:
        # Add your indicator calculations here
        market_data.indicators["sma_20"] = calculate_sma(market_data, 20)
        market_data.indicators["atr"] = calculate_atr(market_data, 14)
        return market_data
    
    def check_entry_conditions(self, market_data: MarketData) -> bool:
        # Implement your entry logic
        return market_data.close > market_data.indicators["sma_20"]
    
    def check_exit_conditions(self, market_data: MarketData) -> bool:
        # Implement your exit logic
        return market_data.close < market_data.indicators["sma_20"]
