# src/trading_strategy/models/market_data.py
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel

class MarketData(BaseModel):
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    indicators: Dict[str, float] = {}

    def calculate_range(self) -> float:
        """Calculate the price range for the period."""
        return self.high - self.low

    def is_bullish(self) -> bool:
        """Determine if the candle is bullish."""
        return self.close > self.open