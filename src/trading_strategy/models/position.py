from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class Position(BaseModel):
    entry_price: float
    size: float
    side: PositionSide
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None 