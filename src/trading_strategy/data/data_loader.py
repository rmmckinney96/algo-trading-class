import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta

from trading_strategy.models.market_data import MarketData
from trading_strategy.models.instrument import Instrument

class MarketDataLoader:
    @classmethod
    def load_historical_data(
        cls, 
        instrument: Instrument, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        interval: str = '1d'
    ) -> List[MarketData]:
        """
        Load historical market data using Yahoo Finance
        
        :param instrument: Instrument to fetch data for
        :param start_date: Start date for data (default: 1 year ago)
        :param end_date: End date for data (default: today)
        :param interval: Data interval (1d, 1h, etc.)
        :return: List of MarketData objects
        """
        # Set default dates if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        # Fetch data
        ticker = yf.Ticker(instrument.symbol.replace('/', '-'))
        df = ticker.history(
            start=start_date, 
            end=end_date, 
            interval=interval
        )
        
        # Convert to MarketData objects
        market_data = []
        for idx, row in df.iterrows():
            market_data.append(MarketData(
                timestamp=idx,
                symbol=instrument.symbol,
                open_price=row['Open'],
                close_price=row['Close'],
                high_price=row['High'],
                low_price=row['Low'],
                volume=row['Volume']
            ))
        
        return market_data

    @classmethod
    def calculate_additional_indicators(
        cls, 
        market_data: List[MarketData], 
        atr_window: int = 14
    ) -> List[MarketData]:
        """
        Calculate additional market indicators like ATR
        
        :param market_data: List of market data
        :param atr_window: Window for Average True Range calculation
        :return: Market data with additional indicators
        """
        # Convert to DataFrame for calculation
        df = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'high': md.high_price,
                'low': md.low_price,
                'close': md.close_price
            } for md in market_data
        ])
        
        # Calculate True Range
        df['tr'] = pd.DataFrame([
            max(
                df['high'].iloc[i] - df['low'].iloc[i],
                abs(df['high'].iloc[i] - df['close'].iloc[i-1]) if i > 0 else 0,
                abs(df['low'].iloc[i] - df['close'].iloc[i-1]) if i > 0 else 0
            ) for i in range(len(df))
        ])
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=atr_window).mean()
        
        # Update market data with ATR
        for i, md in enumerate(market_data):
            md.atr = df['atr'].iloc[i]
        
        return market_data
