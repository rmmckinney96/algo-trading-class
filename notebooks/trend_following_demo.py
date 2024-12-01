#%% [markdown]
# # Trend Following Strategy Demo

#
# This file is configured to run in VS Code's Interactive Window.

# ## Load and prepare market data
#%%
import pandas as pd
from trading_strategy.models.backtest import BacktestRunner
from trading_strategy.models.strategy_config import StrategyConfig, RiskConfig
from trading_strategy.models.market_data import MarketData
from trading_strategy.visualization.performance import PerformanceVisualizer

df = pd.read_csv('../data/raw/USDJPY.csv')
market_data = [
    MarketData(
        timestamp=pd.to_datetime(row['local_time_GMT']),
        symbol='USDJPY',
        open=row['USDJPY_Open'],
        high=row['USDJPY_High'],
        low=row['USDJPY_Low'],
        close=row['USDJPY_Close'],
        volume=None
    ) for _, row in df.iterrows()
]

#%% [markdown]
# ## Configure and run backtest

#%%
strategy_config = StrategyConfig(
    initial_capital=100000,
    risk_config=RiskConfig(
        max_position_risk=0.01,
        max_drawdown=0.10,
        waiting_period_hours=72,
        max_weekly_trades=5,
        trailing_stop_pct=0.02
    ),
    symbol='USDJPY',
    indicators=['sma_20', 'sma_50', 'atr']
)

runner = BacktestRunner(strategy_config=strategy_config)
results = runner.run(market_data)

# Visualize results
visualizer = PerformanceVisualizer(results)
visualizer.print_summary()
visualizer.plot_summary()
# %%
