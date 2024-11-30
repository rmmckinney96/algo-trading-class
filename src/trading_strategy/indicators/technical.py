import numpy as np
from typing import List
from trading_strategy.models.market_data import MarketData

def calculate_sma(data_points: List[MarketData], period: int) -> float:
    """
    Calculate Simple Moving Average
    
    Args:
        data_points: List of market data points
        period: Period for SMA calculation
        
    Returns:
        float: SMA value
    """
    if len(data_points) < period:
        return None
        
    prices = [point.close for point in data_points[-period:]]
    return np.mean(prices)

def calculate_atr(data_points: List[MarketData], period: int) -> float:
    """
    Calculate Average True Range
    
    Args:
        data_points: List of market data points
        period: Period for ATR calculation
        
    Returns:
        float: ATR value
    """
    if len(data_points) < 2:
        return None
        
    true_ranges = []
    for i in range(1, len(data_points)):
        high = data_points[i].high
        low = data_points[i].low
        prev_close = data_points[i-1].close
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    if len(true_ranges) < period:
        return None
        
    return np.mean(true_ranges[-period:]) 