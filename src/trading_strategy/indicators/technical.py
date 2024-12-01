from typing import List, Dict, Any
import numpy as np
from .registry import IndicatorRegistry
from trading_strategy.models.market_data import MarketData

def calculate_indicators_vectorized(data_points: List[MarketData], indicator_configs: Dict[str, Any]) -> List[dict]:
    """Vectorized calculation of multiple indicators"""
    # Convert to numpy arrays once
    prices = np.array([point.close for point in data_points])
    highs = np.array([point.high for point in data_points])
    lows = np.array([point.low for point in data_points])
    
    results = []
    n = len(prices)
    
    # Calculate each indicator using registry
    indicator_values = {}
    for name, config in indicator_configs.items():
        # Parse indicator name and period
        if '_' in name:
            base_name, period = name.rsplit('_', 1)
            period = int(period)
        else:
            base_name, period = name, config.get('period', 14)
            
        # Get indicator function from registry
        indicator_func = IndicatorRegistry.get_indicator(base_name)
        if indicator_func:
            if base_name == 'atr':
                values = indicator_func(highs, lows, prices, period)
            else:
                values = indicator_func(prices, period)
            indicator_values[name] = values
    
    # Combine results
    for i in range(n):
        indicators = {}
        for name, values in indicator_values.items():
            indicators[name] = float(values[i]) if not np.isnan(values[i]) else None
        results.append(indicators)
    
    return results 