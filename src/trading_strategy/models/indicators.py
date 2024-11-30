from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TechnicalIndicators(BaseModel):
    timestamp: datetime
    ma_short: Optional[float] = None
    ma_medium: Optional[float] = None
    ma_long: Optional[float] = None
    atr: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

class CustomIndicators(TechnicalIndicators):
    """Extend this class to add custom indicators"""
    pass 