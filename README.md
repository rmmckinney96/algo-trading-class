# Trading Strategy Framework

## 🚀 Overview

The Trading Strategy Framework is a flexible, modular Python library designed to simplify the development, backtesting, and analysis of trading strategies across various financial instruments.

## ✨ Features

- **Flexible Architecture**
  - Modular design for easy strategy development
  - Supports multiple asset classes (Forex, Stocks, Crypto)
  - Comprehensive risk management

- **Advanced Modeling**
  - Pydantic-based data models
  - Type-safe configuration
  - Detailed performance tracking

- **Key Components**
  - Instrument modeling
  - Market data handling
  - Strategy configuration
  - Performance metrics
  - Risk management utilities

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-strategy-framework.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -e .
```

## 📊 Quick Start

### Creating a Strategy

```python
from trading_strategy.models.instrument import Instrument
from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.strategies.trend_following import TrendFollowingStrategy
from trading_strategy.data.data_loader import MarketDataLoader

# Define Instrument
usd_jpy = Instrument(
    symbol="USD/JPY", 
    base_currency="USD", 
    quote_currency="JPY"
)

# Configure Strategy
config = StrategyConfig(
    name="Basic Trend Following",
    initial_capital=10000,
    max_position_risk=0.02
)

# Load Market Data
market_data = MarketDataLoader.load_historical_data(usd_jpy)

# Initialize Strategy
strategy = TrendFollowingStrategy(usd_jpy, config)

# Process Market Data
for data_point in market_data:
    strategy.process_market_data(data_point)

# Analyze Performance
print(strategy.performance)
```

## 🧠 Key Concepts

### 1. Instrument Modeling
- Supports multiple asset types
- Flexible symbol and currency definitions

### 2. Strategy Configuration
- Risk management parameters
- Capital allocation rules
- Trade frequency controls

### 3. Performance Tracking
- Detailed trade logs
- Equity curve generation
- Comprehensive performance metrics

## 📈 Supported Strategy Types

- Trend Following
- Mean Reversion
- Breakout
- Statistical Arbitrage (Planned)

## 🔧 Development

### Testing
```bash
# Run tests
pytest tests/

# Run type checking
mypy src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📋 Roadmap

- [ ] Additional strategy templates
- [ ] Machine learning integration
- [ ] Advanced backtesting suite
- [ ] Live trading module
- [ ] More comprehensive documentation

## 📜 License

MIT License

## 🤝 Support

For questions, issues, or feature requests, please open a GitHub issue.

## 📚 References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [TA-Lib Technical Indicators](https://ta-lib.org/)

---

**Disclaimer**: This framework is for educational purposes. Always perform thorough testing and understand the risks before using in live trading.



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

