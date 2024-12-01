import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from trading_strategy.models.backtest import BacktestResults

class PerformanceVisualizer:
    def __init__(self, results: BacktestResults):
        self.results = results
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare DataFrames for visualization"""
        # Prepare equity data
        self.equity_df = pd.DataFrame([(p.timestamp, p.equity, p.unrealized_pnl) 
                                     for p in self.results.equity_curve],
                                    columns=['timestamp', 'equity', 'unrealized_pnl'])
        
        # Prepare trade data
        self.trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'pnl': t.pnl,
            'return': (t.pnl / (t.entry_price * abs(t.size))) if t.size != 0 else 0
        } for t in self.results.trades])
        
        # Calculate drawdowns
        self.drawdowns = (self.equity_df['equity'].cummax() - self.equity_df['equity']) / self.equity_df['equity'].cummax() * 100
        
        # Calculate monthly returns
        self.monthly_returns = pd.DataFrame({
            'equity': self.equity_df.set_index('timestamp')['equity']
        }).resample('M').last().pct_change() * 100

    def plot_summary(self):
        """Create summary performance charts"""
        #plt.style.use('seaborn')
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        
        # Plot equity curve (log scale)
        axes[0].plot(self.equity_df['timestamp'], self.equity_df['equity'])
        axes[0].set_title('Equity Curve (Log Scale)')
        axes[0].set_yscale('log')
        axes[0].grid(True)
        axes[0].set_ylabel('Equity ($)')
        
        # Plot drawdown
        axes[1].fill_between(self.equity_df['timestamp'], self.drawdowns, color='red', alpha=0.3)
        axes[1].set_title('Drawdown (%)')
        axes[1].grid(True)
        axes[1].set_ylabel('Drawdown %')
        
        # Plot trade PnL distribution
        self.trades_df['pnl'].hist(bins=50, ax=axes[2])
        axes[2].set_title('Trade PnL Distribution')
        axes[2].grid(True)
        axes[2].set_xlabel('PnL ($)')
        axes[2].set_ylabel('Frequency')
        
        # Plot cumulative trade returns
        self.trades_df['cumulative_return'] = (1 + self.trades_df['return']).cumprod() - 1
        axes[3].plot(self.trades_df['exit_time'], self.trades_df['cumulative_return'] * 100)
        axes[3].set_title('Cumulative Trade Returns (%)')
        axes[3].grid(True)
        axes[3].set_ylabel('Return %')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_monthly_returns(self):
        """Plot monthly return distribution"""
        fig, ax = plt.subplots(figsize=(15, 5))
        self.monthly_returns['equity'].plot(kind='bar', ax=ax)
        ax.set_title('Monthly Returns (%)')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig, ax
    
    def print_summary(self):
        """Print performance summary"""
        print("Performance Metrics:")
        print(f"Total Return: {self.results.total_return_pct:.2f}%")
        print(f"Annualized Return: {self.results.annualized_return:.2f}%")
        print(f"Max Drawdown: {self.results.max_drawdown_pct:.2f}%")
        
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {self.results.metrics.total_trades}")
        print(f"Win Rate: {self.results.metrics.win_rate:.2f}%")
        print(f"Profit Factor: {self.results.metrics.profit_factor:.2f}")
        print(f"Average Trade PnL: ${self.results.metrics.avg_trade_pnl:.2f}")
        print(f"Average Trade Return: {self.results.metrics.avg_trade_return:.2f}%")
        print(f"Average Trade Duration: {self.results.metrics.avg_duration}")
        print(f"Sharpe Ratio: {self.results.metrics.sharpe_ratio:.2f}")
        print(f"Average Daily Return: {self.results.metrics.avg_daily_return:.2f}%")
        
        print("\nMonthly Return Statistics:")
        print(f"Average Monthly Return: {self.monthly_returns['equity'].mean():.2f}%")
        print(f"Monthly Return Std: {self.monthly_returns['equity'].std():.2f}%")
        print(f"Best Month: {self.monthly_returns['equity'].max():.2f}%")
        print(f"Worst Month: {self.monthly_returns['equity'].min():.2f}%") 