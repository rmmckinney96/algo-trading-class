import pandas as pd
import matplotlib.pyplot as plt
from risk_manager import RiskManager
from forex_calculator import ForexCalculator

class Tradingself:

    def check_market_conditions(self, row):
        raise NotImplementedError()
    
    def check_entry_conditions(self, row):
        raise NotImplementedError()
        
    def check_exit_conditions(self, row, current_price):
        raise NotImplementedError()
    
    def _open_position(self, price, date, row):
        raise NotImplementedError()
    
    def _close_position(self, price, date, reason):
        raise NotImplementedError()
    
    def manage_position(self, current_price, current_date, row):
        if current_price > self.highest_price:
            self.highest_price = current_price
            self.trailing_stop_price = self.forex_calculator.calculate_trailing_stop_price(
                self.highest_price, TRAILING_STOP_PERCENTAGE
            )

        should_exit, exit_reason = self.check_exit_conditions(row, current_price)
        if should_exit:
            self._close_position(current_price, current_date, exit_reason)

    def calculate_current_equity(self, current_price):
        if self.current_position > 0:
            self.unrealized_pnl = self.forex_calculator.calculate_profit_usd(
                self.entry_price, current_price, self.current_position
            )
            self.current_equity = self.capital + self.unrealized_pnl
        else:
            self.unrealized_pnl = 0
            self.current_equity = self.capital
            
        return self.current_equity
        
    def calculate_position_size(self, current_price, current_atr):
        risk_amount = self.capital * self.max_position_risk
        position_size = (risk_amount / (current_atr * 3.0)) * current_price
        return min(position_size, (self.capital * LEVERAGE) / current_price)
    
    def check_waiting_period(self, current_time):
        if self.halt_start_time is None:
            return True
        
        time_difference = current_time - self.halt_start_time
        hours_passed = time_difference.total_seconds() / 3600
        return hours_passed >= self.waiting_period_hours
    
    def execute_trade(self, data):
        for i in range(1, len(data)):
            self.current_index = i
            current_price = data.iloc[i]['USDJPY_Close']
            current_date = data.iloc[i]['local_time_GMT']
            
            current_equity = self.calculate_current_equity(current_price)
            
            self.equity_curve.append({
                'Date': current_date,
                'Equity': current_equity,
                'Unrealized_PnL': self.unrealized_pnl
            })
            
            self.risk_manager.update_peak_capital(current_equity)
            
            if current_equity < self.risk_manager.get_minimum_equity():
                if self.current_position > 0:
                    self._close_position(current_price, current_date, 'Drawdown Protection')
                    self.risk_manager.reset_peak_capital(current_equity)
                    self.trading_halted = True
                continue
            
            if i > 0 and data.iloc[i]['Week_Number'] != data.iloc[i-1]['Week_Number']:
                self.weekly_trade_count = 0
            
            if self.trading_halted:
                waiting_period_elapsed = self.check_waiting_period(current_date)
                market_conditions_favorable = self.check_market_conditions(data.iloc[i])
                
                if waiting_period_elapsed and market_conditions_favorable:
                    self.trading_halted = False
                    self.consecutive_losses = 0
                    self.halt_start_time = None
                else:
                    if self.current_position > 0:
                        self.manage_position(current_price, current_date, data.iloc[i])
                    continue
            
            if self.current_position > 0:
                self.manage_position(current_price, current_date, data.iloc[i])
                continue
            
            if (self.current_position == 0 and 
                self.weekly_trade_count < WEEKLY_TRADE_LIMIT and 
                self.check_entry_conditions(data.iloc[i])):
                
                self._open_position(current_price, current_date, data.iloc[i])
                self.weekly_trade_count += 1


    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.current_equity = initial_capital
        self.risk_manager = RiskManager(initial_capital)
        self.forex_calculator = ForexCalculator()
        self.trade_log = []
        self.equity_curve = []
        self.current_position = 0
        self.entry_price = None
        self.highest_price = None
        self.trailing_stop_price = None
        self.weekly_trade_count = 0
        self.trading_halted = False
        self.consecutive_losses = 0
        self.current_index = 0
        self.unrealized_pnl = 0
        self.max_position_risk = 0.02
        self.halt_start_time = None
        self.waiting_period_hours = 72


        data = self.prepare_data('USDJPY.csv')
        self.execute_trade(data)
        
        trade_log_df = pd.DataFrame(self.trade_log)
        print("total trades " + str(len(trade_log_df)/2))
        equity_curve_df = pd.DataFrame(self.equity_curve)
        
        # Plotting Equity Curve with left-side legend
        plt.figure(figsize=(15, 7))
        plt.plot(equity_curve_df['Date'], equity_curve_df['Equity'], 
                label='Equity Curve', color='blue', linewidth=2)
        plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', 
                    label='Initial Capital', linewidth=1)
        plt.title('Equity Curve with Enhanced Risk Management')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Price Chart with Signals with left-side legend
        plt.figure(figsize=(15, 7))
        plt.plot(data['local_time_GMT'], data['USDJPY_Close'], 
                label='USD/JPY', color='gray', alpha=0.6)
        plt.plot(data['local_time_GMT'], data['MA_300'], 
                label='300-Period MA', color='blue', alpha=0.7)
        plt.plot(data['local_time_GMT'], data['MA_504'], 
                label='504-Period MA', color='green', alpha=0.7)
        
        buy_signals = trade_log_df[trade_log_df['Type'] == 'Buy']
        sell_signals = trade_log_df[trade_log_df['Type'] == 'Sell']
        
        plt.scatter(buy_signals['Date'], buy_signals['Price'], 
                marker='^', color='g', s=100, label='Buy')
        plt.scatter(sell_signals['Date'], sell_signals['Price'], 
                marker='v', color='r', s=100, label='Sell')
        
        plt.title('USD/JPY Price with Enhanced Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        # Run performance analysis
        analyze_performance(trade_log_df, self, equity_curve_df, data)