from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class Position(BaseModel):
    side: PositionSide
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    metadata: dict = Field(default_factory=dict)

class Trade(Position):
    exit_price: float
    exit_time: datetime
    pnl: float
    exit_reason: str 