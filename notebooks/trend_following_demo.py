#%% [markdown]
# # Trend Following Strategy Demo
# 
# This notebook demonstrates the implementation and results of a basic trend following strategy.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from trading_strategy.models.strategy_config import StrategyConfig, RiskConfig
from trading_strategy.strategies.trend_following import TrendFollowingStrategy
from trading_strategy.models.market_data import MarketData

#%% [markdown]
# ## 1. Load and Prepare Data

#%%
# Load sample data
df = pd.read_csv('../data/raw/USDJPY.csv')

# Convert to MarketData objects
market_data = [
    MarketData(
        timestamp=pd.to_datetime(row['local_time_GMT']),
        symbol='USDJPY',
        open=row['USDJPY_Open'],
        high=row['USDJPY_High'],
        low=row['USDJPY_Low'],
        close=row['USDJPY_Close'],
        volume=None  # Since volume is not in your CSV
    ) for _, row in df.iterrows()
]

#%% [markdown]
# ## 2. Configure Strategy

#%%
# Define risk configuration
risk_config = RiskConfig(
    max_position_risk=0.02,
    max_drawdown=0.10,
    waiting_period_hours=72,
    max_weekly_trades=5,
    trailing_stop_pct=0.02
)

# Create strategy configuration
strategy_config = StrategyConfig(
    initial_capital=100000,
    risk_config=risk_config,
    symbol='USDJPY',
    indicators=['sma_20', 'sma_50', 'atr']
)

# Initialize strategy
strategy = TrendFollowingStrategy(strategy_config)

#%% [markdown]
# ## 3. Run Backtest

#%%
# Process each market data point
equity_curve = []
positions = []

for data in market_data:
    # Calculate indicators
    data_with_indicators = strategy.calculate_indicators(data)
    
    # Check for position exit
    if strategy.position and strategy.check_exit_conditions(data_with_indicators):
        trade = strategy.close_position(data_with_indicators)
        positions.append(trade)
    
    # Check for position entry
    elif not strategy.position and strategy.check_entry_conditions(data_with_indicators):
        position_size = strategy.calculate_position_size(data_with_indicators)
        strategy.open_position(data_with_indicators, position_size)
    
    # Record equity
    equity_curve.append({
        'timestamp': data.timestamp,
        'equity': strategy.current_equity
    })

#%% [markdown]
# ## 4. Analyze Results

#%%
# Convert results to DataFrame
equity_df = pd.DataFrame(equity_curve)
trades_df = pd.DataFrame([vars(trade) for trade in positions])

# Calculate key metrics
total_return = (equity_df['equity'].iloc[-1] / strategy_config.initial_capital - 1) * 100
win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
max_drawdown = (equity_df['equity'].max() - equity_df['equity'].min()) / equity_df['equity'].max() * 100

print(f"Total Return: {total_return:.2f}%")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Total Trades: {len(trades_df)}")

#%% [markdown]
# ## 5. Visualize Results

#%%
plt.figure(figsize=(15, 8))

# Plot equity curve
plt.subplot(2, 1, 1)
plt.plot(equity_df['timestamp'], equity_df['equity'])
plt.title('Equity Curve')
plt.grid(True)

# Plot trade PnL distribution
plt.subplot(2, 1, 2)
trades_df['pnl'].hist(bins=50)
plt.title('Trade PnL Distribution')
plt.grid(True)

plt.tight_layout()
plt.show() 