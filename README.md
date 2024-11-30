# Trading Strategy Framework

## Overview
A flexible, modular trading strategy framework built with Python, featuring:
- Pydantic-based data models
- Extensible strategy design
- Risk management

## Project Structure
```sh
trading_strategy_framework/
│
├── pyproject.toml
├── README.md
│
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── backtesting_analysis.ipynb
│   └── strategy_exploration.ipynb
│
├── scripts/
│   ├── live_trading.py
│   └── run_backtest.py
│
├── src/
│   └── trading_strategy/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base_strategy.py
│       │   └── execution_manager.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── data_loader.py
│       │   └── data_preprocessor.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── instrument.py
│       │   ├── market_data.py
│       │   ├── order.py
│       │   ├── performance.py
│       │   └── strategy_config.py
│       │
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── mean_reversion.py
│       │   └── trend_following.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── performance_analysis.py
│           └── risk_management.py
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_performance.py
│   └── test_strategies.py

```

