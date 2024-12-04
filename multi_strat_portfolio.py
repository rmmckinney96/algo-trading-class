# %% [markdown]
# # Multi-Strategy Portfolio Construction

# %% Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict

# Import our modules
from modules.portfolio import Strategy, KellyPortfolio, EqualWeightPortfolio
from modules.data_loader import load_signals, get_price_data
from modules.visualization import (
    plot_portfolio_metrics,
    plot_strategy_comparison,
    plot_strategy_analysis
)

# %% Load and process data
signal_configs = [
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

costs_config = {
    'transaction_fee_pct': 0.0005,
    'slippage_pct': 0.0002,
    'borrowing_cost_pa': 0.01
}

# Load data
signals = load_signals(signal_configs)
prices = get_price_data(signals['symbol'].unique(), directory='prices/')

# %% Create and run portfolio
portfolio = EqualWeightPortfolio(initial_capital=1000000)

# Create strategies
for strategy_name in signals['strategy'].unique():
    strategy_signals = signals[signals['strategy'] == strategy_name]
    strategy = Strategy(strategy_name, strategy_signals, prices)
    portfolio.add_strategy(strategy, allocation=1/len(signals['strategy'].unique()))

# Calculate returns
returns = portfolio.calculate_returns(costs_config)

# %% Portfolio Analysis
plot_portfolio_metrics(portfolio)
plot_strategy_comparison(portfolio)

# %% Strategy Analysis
for strategy_name in portfolio.strategies:
    plot_strategy_analysis(portfolio, strategy_name)

# %% Current Positions
active_positions = portfolio.positions[
    (portfolio.positions['is_active']) & 
    (portfolio.positions['timestamp'] == portfolio.positions['timestamp'].max())
]
print("\nCurrent Portfolio Positions")
print("==========================")
if not active_positions.empty:
    print(active_positions[['symbol', 'strategy', 'size', 'allocated_capital', 'returns']].to_string())
else:
    print("No open positions")
