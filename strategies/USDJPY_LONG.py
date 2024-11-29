import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from classes.trading_strategy import TradingStrategy

# Constants and Parameters
INITIAL_CAPITAL = 100000
LEVERAGE = 2
TRAILING_STOP_PERCENTAGE = 0.03  # 3%
DRAWDOWN_LIMIT = 0.05         # 6%
WEEKLY_TRADE_LIMIT = 7
ADX_THRESHOLD = 25
RSI_LOWER_LIMIT = 30
RSI_UPPER_LIMIT = 70
CONSECUTIVE_LOSSES_LIMIT = 1


class USDJPY_LONG(TradingStrategy):

    def prepare_data(file_path):
        data = pd.read_csv(file_path)
        data['local_time_GMT'] = pd.to_datetime(data['local_time_GMT'])
        data = data.sort_values(by='local_time_GMT').reset_index(drop=True)
        
        # Calculate moving averages
        data['MA_short'] = data['USDJPY_Close'].rolling(window=15).mean()
        data['MA_medium'] = data['USDJPY_Close'].rolling(window=80).mean()
        data['MA_long'] = data['USDJPY_Close'].rolling(window=150).mean()
        data['MA_300'] = data['USDJPY_Close'].rolling(window=300).mean()
        data['MA_504'] = data['USDJPY_Close'].rolling(window=504).mean()
        
        # Calculate MA slopes using linear regression
        def calculate_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window:
                    slopes.append(np.nan)
                    continue
                y = series.iloc[i-window:i]
                x = np.arange(window)
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope)
            return pd.Series(slopes, index=series.index)
        
        # Calculate slopes for both MAs
        data['MA_300_Slope'] = calculate_slope(data['MA_300'], 48)
        data['MA_504_Slope'] = calculate_slope(data['MA_504'], 48)
        
        # Technical indicators
        adx = ADXIndicator(high=data['USDJPY_High'], 
                        low=data['USDJPY_Low'], 
                        close=data['USDJPY_Close'], 
                        window=14)
        data['ADX'] = adx.adx()
        
        rsi = RSIIndicator(close=data['USDJPY_Close'], window=20)
        data['RSI'] = rsi.rsi()
        
        atr = AverageTrueRange(high=data['USDJPY_High'], 
                            low=data['USDJPY_Low'], 
                            close=data['USDJPY_Close'], 
                            window=14)
        data['ATR'] = atr.average_true_range()
        
        # Compute 20-period rolling average of ATR
        data['ATR_MA20'] = data['ATR'].rolling(window=20).mean()
        
        # Add week number for trade limiting
        data['Week_Number'] = data['local_time_GMT'].dt.isocalendar().week
        
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        return data
    

if __name__ == "__main__":
    USDJPY_LONG(INITIAL_CAPITAL)