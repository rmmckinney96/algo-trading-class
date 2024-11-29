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
        
    def calculate_position_size(self, current_price, current_atr):
        risk_amount = self.capital * self.max_position_risk
        position_size = (risk_amount / (current_atr * 3.0)) * current_price
        return min(position_size, (self.capital * LEVERAGE) / current_price)
    
    def check_entry_conditions(self, row):
        ma_conditions = (
            row['MA_short'] > row['MA_medium'] and 
            row['MA_short'] > row['MA_long'] and
            row['MA_300_Slope'] > 0 and
            row['MA_504_Slope'] > 0
        )
        
        technical_conditions = (
            row['ADX'] > ADX_THRESHOLD and
            RSI_LOWER_LIMIT < row['RSI'] < RSI_UPPER_LIMIT
        )
        
        volatility_condition = row['ATR'] < row['ATR_MA20'] * 1.5
        
        return ma_conditions and technical_conditions and volatility_condition
        # you can check the strength of signal and then decide positioj sizing 
        # if very strong signal; then go big 
        # But if signal is weak; then reduce leevrage or go soft 
        
    
    def check_exit_conditions(self, row, current_price):
        if self.current_position == 0:
            return False, None

        if current_price <= self.trailing_stop_price:
            return True, 'Trailing Stop Hit'

        if row['MA_short'] < row['MA_medium']:
            return True, 'Short MA Crossed Below Medium MA'

        if row['MA_300_Slope'] < 0:
            return True, 'MA300 Slope Turned Negative'

        if row['ATR'] > row['ATR_MA20'] * 2:
            return True, 'Excessive Volatility'

        return False, None
    
    
    def check_waiting_period(self, current_time):
        if self.halt_start_time is None:
            return True
        
        time_difference = current_time - self.halt_start_time
        hours_passed = time_difference.total_seconds() / 3600
        return hours_passed >= self.waiting_period_hours
    
    def check_market_conditions(self, row):
        return (
            row['MA_300_Slope'] > 0 and 
            row['MA_504_Slope'] > 0 and
            row['ADX'] > ADX_THRESHOLD and
            RSI_LOWER_LIMIT < row['RSI'] < RSI_UPPER_LIMIT
        )

    def manage_position(self, current_price, current_date, row):
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.trailing_stop_price = self.forex_calculator.calculate_trailing_stop_price(
                self.highest_price, TRAILING_STOP_PERCENTAGE
            )

        should_exit, exit_reason = self.check_exit_conditions(row, current_price)
        if should_exit:
            self._close_position(current_price, current_date, exit_reason)
            
    
    def _open_position(self, price, date, row):
        self.entry_price = price
        self.current_position = self.calculate_position_size(price, row['ATR'])
        
        self.trailing_stop_price = self.forex_calculator.calculate_trailing_stop_price(
            price, TRAILING_STOP_PERCENTAGE
        )
        self.highest_price = price
        
        self.trade_log.append({
            'Type': 'Buy',
            'Date': date,
            'Price': price,
            'Position_Size': self.current_position,
            'Capital': self.capital,
            'Trailing_Stop': self.trailing_stop_price,
            'ATR': row['ATR']
        })
    
    def _close_position(self, price, date, reason):
        profit = self.forex_calculator.calculate_profit_usd(
            self.entry_price, price, self.current_position
        )
        self.capital += profit

        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= CONSECUTIVE_LOSSES_LIMIT and not self.trading_halted:
            self.trading_halted = True
            self.halt_start_time = date
            reason = f"{reason} (Triggered Trading Halt)"

        self.trade_log.append({
            'Type': 'Sell',
            'Date': date,
            'Price': price,
            'Profit': profit,
            'Capital': self.capital,
            'Exit_Reason': reason
        })

        self.current_position = 0
        self.entry_price = None
        self.highest_price = None
        self.trailing_stop_price = None
        self.unrealized_pnl = 0
    

if __name__ == "__main__":
    USDJPY_LONG(INITIAL_CAPITAL)