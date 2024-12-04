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
from modules.portfolio import Strategy, Portfolio
from modules.data_loader import load_signals, get_price_data
from modules.visualization import (
    plot_strategy_comparison, 
    plot_portfolio_metrics,
    plot_strategy_analysis,
    plot_drawdowns
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# %% Example usage:
signal_configs = [
    {
        'file_path': "trade_logs/all_trading_logs.csv",
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': {'from_column': 'Strategy'},
            'symbol': 'USA500.IDXUSD',
            'signal': {
                'map_from': 'Action',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    },
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

# %% Load and process data
# Load signals
signals = load_signals(signal_configs)

# Get unique symbols
unique_symbols = signals['symbol'].unique()
print("\nSymbols from signals:")
for symbol in unique_symbols:
    print(f"  - {symbol}")

# Load price data
price_data = get_price_data(signals['symbol'].unique(), directory='prices/')

# Print debug information
print("\nDebug information:")
for symbol in price_data:
    print(f"\n{symbol} price data:")
    print(f"Index timezone: {price_data[symbol].index.tz}")
    print(f"First timestamp: {price_data[symbol].index[0]}")
    print(f"Last timestamp: {price_data[symbol].index[-1]}")

print("\nSignals information:")
print(f"First timestamp: {signals['timestamp'].min()}")
print(f"Last timestamp: {signals['timestamp'].max()}")
print(f"Timestamp dtype: {signals['timestamp'].dtype}")

# %% Create and run portfolio
# Initialize portfolio
portfolio = Portfolio(initial_capital=1000000)

# Create strategies from signals
for strategy_name in signals['strategy'].unique():
    strategy_signals = signals[signals['strategy'] == strategy_name]
    strategy = Strategy(strategy_name, strategy_signals)
    portfolio.add_strategy(strategy, allocation=1/len(signals['strategy'].unique()))

# Calculate returns
portfolio_returns = portfolio.calculate_returns(price_data, costs_config)

# %% Portfolio Overview
print("\nPortfolio Overview")
print("-----------------")
plot_strategy_comparison(portfolio)
plot_portfolio_metrics(portfolio)
plot_drawdowns(portfolio)

# %% Individual Strategy Analysis
print("\nIndividual Strategy Analysis")
print("-------------------------")
for strategy_name in signals['strategy'].unique():
    print(f"\nAnalyzing {strategy_name}")
    plot_strategy_analysis(
        portfolio,
        strategy_name=strategy_name,
        start_date=signals['timestamp'].min(),
        end_date=signals['timestamp'].max()
    )

# %% Time Period Analysis
# Example: Last 6 months analysis
end_date = signals['timestamp'].max()
start_date = end_date - pd.Timedelta(days=180)

print("\nLast 6 Months Analysis")
print("--------------------")
plot_strategy_analysis(
    portfolio,
    start_date=start_date,
    end_date=end_date
)

# %%
