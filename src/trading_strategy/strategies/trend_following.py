from typing import List
from trading_strategy.core.strategy import TradingStrategy
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.position import Position, PositionSide
from trading_strategy.indicators.technical import calculate_sma, calculate_atr

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.price_history: List[MarketData] = []
    
    def calculate_indicators(self, market_data: MarketData) -> MarketData:
        # Store price history for indicator calculations
        self.price_history.append(market_data)
        
        # Calculate indicators using price history
        market_data.indicators["sma_20"] = calculate_sma(self.price_history, 20)
        market_data.indicators["sma_50"] = calculate_sma(self.price_history, 50)
        market_data.indicators["atr"] = calculate_atr(self.price_history, 14)
        
        # Update equity curve with unrealized PnL if we have an open position
        if self.position:
            if self.position.side == PositionSide.LONG:
                unrealized_pnl = (market_data.close - self.position.entry_price) * self.position.size
            else:
                unrealized_pnl = (self.position.entry_price - market_data.close) * self.position.size
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
