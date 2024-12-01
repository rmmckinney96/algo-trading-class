from typing import List, Dict, Union
import numpy as np
from .registry import IndicatorRegistry

@IndicatorRegistry.register('sma')
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Vectorized SMA calculation"""
    weights = np.ones(period) / period
    sma = np.convolve(prices, weights, mode='valid')
    return np.pad(sma, (period-1, 0), mode='constant', constant_values=np.nan)

@IndicatorRegistry.register('atr')
def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Vectorized ATR calculation"""
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1])
    ])
    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    return np.pad(atr, (period, 0), mode='constant', constant_values=np.nan)

@IndicatorRegistry.register('rsi')
def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Vectorized RSI calculation"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed > 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[period] = 100 - 100/(1+rs)
    
    for i in range(period+1, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down if down != 0 else 0
        rsi[i] = 100 - 100/(1+rs)
    
    return rsi 