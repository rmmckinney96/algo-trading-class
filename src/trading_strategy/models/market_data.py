# src/trading_strategy/models/market_data.py
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class MarketData(BaseModel):
    timestamp: datetime = Field(..., description="Market data timestamp")
    symbol: str = Field(..., description="Trading symbol")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: Optional[float] = Field(None, description="Trading volume")
    indicators: Dict[str, float] = Field(default_factory=dict, description="Technical indicators")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def calculate_range(self) -> float:
        """Calculate the price range for the period."""
        return self.high - self.low

    def is_bullish(self) -> bool:
        """Determine if the candle is bullish."""
        return self.close > self.open