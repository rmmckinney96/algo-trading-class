from typing import Callable, Dict, List, Type
from functools import wraps
import numpy as np
from trading_strategy.models.market_data import MarketData

class IndicatorRegistry:
    """Registry for technical indicators"""
    _indicators: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """Decorator to register an indicator calculation function"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Use function name if no name provided
            indicator_name = name or func.__name__
            cls._indicators[indicator_name] = wrapper
            return wrapper
        return decorator
    
    @classmethod
    def get_indicator(cls, name: str) -> Callable:
        """Get indicator calculation function by name"""
        return cls._indicators.get(name)
    
    @classmethod
    def list_indicators(cls) -> List[str]:
        """List all registered indicators"""
        return list(cls._indicators.keys()) 