# %% [markdown]
# # Multi-Strategy Portfolio
# %%

from modules.data_loader import load_signals, get_price_data
from modules.portfolio import (calculate_strategy_returns, 
                             calculate_portfolio_returns,
                             calculate_risk_metrics)
from modules.visualization import (plot_cumulative_returns,
                                 plot_correlation_matrix,
                                 plot_rolling_volatility,
                                 plot_weights_history)
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Signal configuration
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

# Load and process data
signals = load_signals(signal_configs, SCRIPT_DIR)
price_data = get_price_data(signals['symbol'].unique(), SCRIPT_DIR, directory='prices/')
strategy_returns = calculate_strategy_returns(signals, price_data)

# Calculate portfolio returns and metrics
portfolio_values, weights_history, rebalance_dates = calculate_portfolio_returns(
    strategy_returns, 
    rebalance_threshold=0.05
)

# Calculate and display risk metrics
risk_metrics = pd.DataFrame()
for col in strategy_returns.columns:
    metrics = calculate_risk_metrics(strategy_returns[col], col)
    risk_metrics = pd.concat([risk_metrics, metrics], axis=1)

portfolio_metrics = calculate_risk_metrics(portfolio_values['returns'], 'Portfolio')
risk_metrics = pd.concat([risk_metrics, portfolio_metrics], axis=1)

print("\nRisk Metrics Summary:")
print(risk_metrics.round(4))

# Create and display plots
plt.style.use('default')

fig1 = plot_cumulative_returns(strategy_returns, portfolio_values['returns'])
fig2 = plot_correlation_matrix(strategy_returns)
fig3 = plot_rolling_volatility(strategy_returns)
fig4 = plot_weights_history(weights_history)

plt.show()
# %%
