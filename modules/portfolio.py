import pandas as pd
import numpy as np
from typing import Dict, Optional

class Strategy:
    def __init__(self, name: str, signals: pd.DataFrame):
        """
        Initialize strategy with signals DataFrame
        
        Parameters:
        -----------
        name : str
            Strategy name
        signals : pd.DataFrame
            DataFrame with columns: timestamp, symbol, signal
        """
        self.name = name
        self.signals = signals.set_index('timestamp').copy()
        self.positions = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.allocated_capital = 0.0
        
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], costs_config: Dict) -> pd.Series:
        """
        Calculate strategy returns using price data and signals
        
        Parameters:
        -----------
        price_data : Dict[str, pd.DataFrame]
            Dictionary of price DataFrames for each symbol
        costs_config : Dict
            Trading cost parameters
        """
        all_returns = []
        
        for symbol in self.signals['symbol'].unique():
            if symbol not in price_data:
                continue
            
            # Get symbol-specific data
            symbol_signals = self.signals[self.signals['symbol'] == symbol]['signal']
            prices = price_data[symbol]
            
            # Align signals with price data
            positions = pd.DataFrame(index=prices.index)
            positions['signal'] = symbol_signals.reindex(prices.index).ffill().fillna(0)
            positions['price'] = prices['Close']
            positions['returns'] = prices['returns']
            
            # Calculate trading costs
            position_changes = positions['signal'].diff().fillna(0)
            trading_mask = position_changes != 0
            
            # Entry and exit prices for cost calculation
            positions['entry_price'] = positions['price'].where(trading_mask).ffill()
            positions['position_return'] = (positions['price'] / positions['entry_price'] - 1) * positions['signal']
            
            # Transaction costs including P&L
            transaction_costs = pd.Series(0.0, index=positions.index)
            transaction_costs[trading_mask] = (
                abs(positions['signal'][trading_mask]) * 
                (1 + positions['position_return'][trading_mask]) * 
                (costs_config['transaction_fee_pct'] + costs_config['slippage_pct'])
            )
            
            # Borrowing costs for short positions
            borrowing_costs = pd.Series(0.0, index=positions.index)
            borrowing_costs[positions['signal'] < 0] = (
                abs(positions['signal']) * 
                (1 + positions['position_return']) * 
                costs_config['borrowing_cost_pa'] / (24 * 252)  # Hourly rate
            )
            
            # Calculate returns after costs
            symbol_returns = (positions['signal'] * positions['returns'] - 
                            transaction_costs - borrowing_costs)
            
            all_returns.append(symbol_returns)
        
        # Combine returns across symbols
        if all_returns:
            strategy_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        else:
            strategy_returns = pd.Series(0, index=self.signals.index)
            
        return strategy_returns

class Portfolio:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.returns = pd.DataFrame()
        
    def add_strategy(self, strategy: Strategy, allocation: float):
        """Add strategy with initial allocation"""
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")
        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        strategy.allocated_capital = self.current_capital * allocation
        
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict) -> pd.DataFrame:
        """Calculate returns for all strategies and portfolio"""
        strategy_returns = {}
        
        # Calculate returns for each strategy
        for name, strategy in self.strategies.items():
            strategy_returns[name] = strategy.calculate_returns(price_data, costs_config)
        
        # Combine into DataFrame
        self.returns = pd.DataFrame(strategy_returns)
        
        # Calculate portfolio returns
        weights = pd.Series(self.allocations)
        self.returns['Portfolio'] = self.returns.mul(weights).sum(axis=1)
        
        return self.returns