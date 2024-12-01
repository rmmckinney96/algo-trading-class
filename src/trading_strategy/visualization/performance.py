import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from trading_strategy.models.backtest import BacktestResults

class PerformanceVisualizer:
    def __init__(self, results: BacktestResults):
        self.results = results
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare DataFrames for visualization - more efficient data preparation"""
        # Convert market data to DataFrame once
        self.market_df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'close': d.close
        } for d in self.results.market_data]).set_index('timestamp')
        
        # Convert equity curve to DataFrame once
        self.equity_df = pd.DataFrame([{
            'timestamp': p.timestamp,
            'equity': p.equity,
            'unrealized_pnl': p.unrealized_pnl
        } for p in self.results.equity_curve]).set_index('timestamp')
        
        # Calculate buy & hold performance once
        initial_price = self.market_df['close'].iloc[0]
        self.buy_hold_equity = self.market_df['close'].div(initial_price).mul(
            self.results.strategy_config.initial_capital
        )
        
        # Calculate drawdowns vectorized
        self.drawdowns = (
            self.equity_df['equity'].cummax() - self.equity_df['equity']
        ).div(self.equity_df['equity'].cummax()).mul(100)
        
        # Prepare trade data efficiently
        if self.results.trades:
            self.trades_df = pd.DataFrame([{
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'pnl': t.pnl,
                'return': (t.pnl / (t.entry_price * abs(t.size))) if t.size != 0 else 0
            } for t in self.results.trades])
            
            # Calculate cumulative returns vectorized
            self.trades_df['cumulative_return'] = (
                (1 + self.trades_df['return']).cumprod() - 1
            ).mul(100)
        
        # Calculate monthly returns vectorized
        self.monthly_returns = self.equity_df['equity'].resample('ME').last().pct_change().mul(100)

    def plot_summary(self):
        """Create summary performance charts - more efficient plotting"""
        fig = plt.figure(figsize=(15, 25))
        gs = plt.GridSpec(6, 2, figure=fig)
        
        # Equity curve (log scale)
        ax1 = fig.add_subplot(gs[0, :])
        self.equity_df['equity'].plot(ax=ax1, logy=True)
        ax1.set_title('Equity Curve (Log Scale)')
        ax1.grid(True)
        ax1.set_ylabel('Equity ($)')
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        self.drawdowns.plot(ax=ax2, color='red', alpha=0.3, kind='area')
        ax2.set_title('Drawdown (%)')
        ax2.grid(True)
        ax2.set_ylabel('Drawdown %')
        
        # Strategy vs Buy & Hold
        ax3 = fig.add_subplot(gs[2, :])
        self.equity_df['equity'].plot(ax=ax3, label='Strategy')
        self.buy_hold_equity.plot(ax=ax3, label='Buy & Hold')
        ax3.set_title('Strategy vs Buy & Hold Performance')
        ax3.set_ylabel('Equity')
        ax3.legend()
        
        # Trade PnL distribution
        ax4 = fig.add_subplot(gs[3, 0])
        if hasattr(self, 'trades_df'):
            self.trades_df['pnl'].hist(bins=50, ax=ax4)
        ax4.set_title('Trade PnL Distribution')
        ax4.grid(True)
        ax4.set_xlabel('PnL ($)')
        ax4.set_ylabel('Frequency')
        
        # Monthly returns
        ax5 = fig.add_subplot(gs[3, 1])
        self.monthly_returns.plot(kind='bar', ax=ax5)
        ax5.set_title('Monthly Returns (%)')
        ax5.grid(True)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # Cumulative trade returns
        ax6 = fig.add_subplot(gs[4:, :])
        if hasattr(self, 'trades_df'):
            self.trades_df.set_index('exit_time')['cumulative_return'].plot(ax=ax6)
        ax6.set_title('Cumulative Trade Returns (%)')
        ax6.grid(True)
        ax6.set_ylabel('Return %')
        
        plt.tight_layout()
        return fig
    
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
        print(f"Average Monthly Return: {self.monthly_returns.mean():.2f}%")
        print(f"Monthly Return Std: {self.monthly_returns.std():.2f}%")
        print(f"Best Month: {self.monthly_returns.max():.2f}%")
        print(f"Worst Month: {self.monthly_returns.min():.2f}%")