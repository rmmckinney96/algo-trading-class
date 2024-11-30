from typing import Literal
from pydantic import BaseModel, Field

class Instrument(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., USD/JPY)")
    base_currency: str = Field(..., description="Base currency")
    quote_currency: str = Field(..., description="Quote currency")
    type: Literal['forex', 'crypto', 'equity'] = Field('forex', description="Instrument type")
    
    @property
    def full_name(self) -> str:
        """Return the full name of the instrument."""
        return f"{self.base_currency}/{self.quote_currency}"