# %% [markdown]
# # Multi-Strategy Portfolio Construction

# %% Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict, Optional

# Import our modules
from modules.portfolio import Strategy, KellyPortfolio, EqualWeightPortfolio
from modules.data_loader import load_signals, get_price_data
from modules.visualization import (
    plot_strategy_comparison, 
    plot_portfolio_metrics,
    plot_strategy_analysis,
    plot_drawdowns
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# %% Load and process data
signal_configs = [
    # {
    #     'file_path': "trade_logs/all_trading_logs.csv",
    #     'mapping_config': {
    #         'timestamp': {'from_column': 'Date'},
    #         'strategy': {'from_column': 'Strategy'},
    #         'symbol': 'USA500.IDXUSD',
    #         'signal': {
    #             'map_from': 'Action',
    #             'values': {'Buy': 1, 'Sell': 0}
    #         }
    #     }
    # },
    {
        'file_path': "trade_logs/trade_log_jupyter.csv",
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': 'Momentum Long-Only',
            'symbol': 'USDJPY',
            'signal': {
                'map_from': 'Type',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    }
]

# Trading costs configuration
costs_config = {
    'transaction_fee_pct': 0.0005,  # 0.05% commission per trade
    'slippage_pct': 0.0002,        # 0.02% slippage
    'borrowing_cost_pa': 0.01      # 1% annual borrowing cost
}

# Load signals and price data
signals = load_signals(signal_configs)
price_data = get_price_data(signals['symbol'].unique(), directory='prices/')

# %% Create and run portfolio
# Initialize portfolio
portfolio = EqualWeightPortfolio(initial_capital=1000000)  #KellyPortfolio(initial_capital=1000000, lookback_days=30)

# Create strategies from signals
for strategy_name in signals['strategy'].unique():
    strategy_signals = signals[signals['strategy'] == strategy_name]
    strategy = Strategy(strategy_name, strategy_signals)
    portfolio.add_strategy(strategy, allocation=1/len(signals['strategy'].unique()))

# Calculate returns
portfolio_returns = portfolio.calculate_returns(price_data, costs_config)

# %% Portfolio PnL Analysis
print("\nPortfolio Summary")
print("================")
summary = portfolio.get_pnl_summary()
if not summary.empty:
    print(summary.to_string(index=False))
else:
    print("No PnL data available yet")

print("\nStrategy PnL Breakdown")
print("=====================")
for strategy_name in portfolio.strategies:
    strategy_pnl = portfolio.get_strategy_pnl(strategy_name)
    if not strategy_pnl.empty:
        print(f"\n{strategy_name}:")
        print(f"Allocated Capital: ${strategy_pnl['allocated_capital'].iloc[-1]:,.2f}")
        print(f"Realized PnL: ${strategy_pnl['realized_pnl'].iloc[-1]:,.2f}")
        print(f"Unrealized PnL: ${strategy_pnl['unrealized_pnl'].iloc[-1]:,.2f}")
        print(f"Total Costs: ${strategy_pnl['total_costs'].iloc[-1]:,.2f}")
        print(f"Net PnL: ${(strategy_pnl['realized_pnl'].iloc[-1] + strategy_pnl['unrealized_pnl'].iloc[-1] - strategy_pnl['total_costs'].iloc[-1]):,.2f}")
        
        # Get trade history for this strategy
        trades = portfolio.strategies[strategy_name].get_trade_history()
        if not trades.empty:
            print(f"Number of trades: {len(trades)}")
            print(f"Average trade PnL: ${trades['realized_pnl'].mean():,.2f}")
            print(f"Win rate: {(trades['realized_pnl'] > 0).mean():.1%}")
    else:
        print(f"\n{strategy_name}: No trades executed yet")

# %% Plot PnL Evolution
plt.figure(figsize=(12, 6))
portfolio.total_pnl.set_index('timestamp')['portfolio_value'].plot(
    title='Portfolio Value Evolution'
)
plt.grid(True)
plt.ylabel('Portfolio Value ($)')
plt.show()

# Plot strategy capital allocation
plt.figure(figsize=(12, 6))
capital_history = portfolio.capital_history.pivot(
    index='timestamp', 
    columns='strategy', 
    values='allocated_capital'
)
capital_history.plot(
    title='Strategy Capital Allocation',
    stacked=True
)
plt.grid(True)
plt.ylabel('Allocated Capital ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot cumulative PnL by strategy
plt.figure(figsize=(12, 6))
for strategy_name in portfolio.strategies:
    strategy_pnl = portfolio.get_strategy_pnl(strategy_name)
    net_pnl = strategy_pnl['realized_pnl'] + strategy_pnl['unrealized_pnl'] - strategy_pnl['total_costs']
    net_pnl.plot(label=strategy_name)
plt.title('Cumulative Strategy PnL')
plt.grid(True)
plt.ylabel('PnL ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% Individual Strategy Analysis
for strategy_name in portfolio.strategies:
    print(f"\nDetailed Analysis: {strategy_name}")
    print("=" * (18 + len(strategy_name)))
    
    strategy = portfolio.strategies[strategy_name]
    trades = strategy.get_trade_history()
    
    if not trades.empty:
        trades['duration'] = trades['exit_time'] - trades['entry_time']
        trades['return'] = trades['realized_pnl'] / (trades['entry_price'] * abs(trades['size']))
        
        print("\nTrade Statistics:")
        print(f"Total trades: {len(trades)}")
        print(f"Win rate: {(trades['realized_pnl'] > 0).mean():.1%}")
        print(f"Average trade duration: {trades['duration'].mean()}")
        print(f"Average return per trade: {trades['return'].mean():.2%}")
        print(f"Best trade: ${trades['realized_pnl'].max():,.2f}")
        print(f"Worst trade: ${trades['realized_pnl'].min():,.2f}")
        
        # Plot trade PnL distribution
        plt.figure(figsize=(10, 5))
        trades['realized_pnl'].hist(bins=50)
        plt.title(f'{strategy_name} - Trade PnL Distribution')
        plt.xlabel('PnL ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# %% Portfolio Analytics
# Calculate key performance metrics
returns = portfolio.total_pnl.set_index('timestamp')['portfolio_value'].pct_change()
cumulative_returns = (1 + returns).cumprod()
drawdowns = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()

print("\nPortfolio Performance Metrics")
print("============================")
print(f"Total Return: {(cumulative_returns.iloc[-1] - 1):.2%}")
print(f"Annualized Return: {(((1 + (cumulative_returns.iloc[-1] - 1)) ** (252/len(returns))) - 1):.2%}")
print(f"Annualized Volatility: {returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")
print(f"Max Drawdown: {drawdowns.min():.2%}")
print(f"Current Drawdown: {drawdowns.iloc[-1]:.2%}")

# Plot portfolio metrics
plot_portfolio_metrics(portfolio)

# Plot drawdown analysis
plot_drawdowns(portfolio)

# Compare strategy performance
plot_strategy_comparison(portfolio)

# %% Strategy Analysis
for strategy_name in portfolio.strategies:
    print(f"\nStrategy Analysis: {strategy_name}")
    print("=" * (len(strategy_name) + 18))
    
    # Get strategy-specific metrics
    plot_strategy_analysis(portfolio, strategy_name)
    
    strategy_pnl = portfolio.get_strategy_pnl(strategy_name)
    strategy_returns = (strategy_pnl['realized_pnl'] + strategy_pnl['unrealized_pnl'] - strategy_pnl['total_costs']).pct_change()
    
    if not strategy_returns.empty:
        print("\nPerformance Metrics:")
        print(f"Annualized Return: {(((1 + strategy_returns.sum()) ** (252/len(strategy_returns))) - 1):.2%}")
        print(f"Annualized Volatility: {strategy_returns.std() * np.sqrt(252):.2%}")
        print(f"Sharpe Ratio: {(strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252):.2f}")
        
        # Calculate rolling metrics
        rolling_sharpe = (
            strategy_returns.rolling(window=30)
            .agg(['mean', 'std'])
            .apply(lambda x: x['mean'] / x['std'] * np.sqrt(252))
        )
        
        # Plot rolling Sharpe ratio
        plt.figure(figsize=(12, 4))
        rolling_sharpe.plot(title=f'{strategy_name} - Rolling 30-Day Sharpe Ratio')
        plt.grid(True)
        plt.show()

# %% Portfolio Positions
print("\nCurrent Portfolio Positions")
print("==========================")
positions = portfolio.get_portfolio_positions()
if not positions.empty:
    print(positions.to_string(index=False))
else:
    print("No open positions")

# Optional: Get summary by strategy
if not positions.empty:
    print("\nPosition Summary by Strategy")
    print("===========================")
    strategy_summary = positions.groupby('strategy').agg({
        'current_size': 'count',
        'unrealized_pnl': 'sum',
        'realized_pnl': 'sum',
        'costs': 'sum',
        'net_pnl': 'sum'
    }).round(2)
    print(strategy_summary)

# %%
