# src/trading_strategy/models/market_data.py
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field
from functools import cached_property

class MarketData(BaseModel):
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    indicators: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        keep_untouched = (cached_property,)
    
    @cached_property
    def price_range(self) -> float:
        """Cached price range calculation"""
        return self.high - self.low
    
    @cached_property
    def is_bullish(self) -> bool:
        """Cached bullish check"""
        return self.close > self.open