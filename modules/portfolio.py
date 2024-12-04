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
        self.current_positions = {}  # symbol -> position size
        
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], costs_config: Dict) -> pd.Series:
        """
        Calculate strategy returns using price data and signals
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
            
            # Calculate position sizes based on allocated capital
            notional_value = self.allocated_capital
            positions['position_size'] = positions['signal'] * notional_value / positions['price']
            
            # Track current positions
            self.current_positions[symbol] = positions['position_size'].iloc[-1]
            
            # Calculate trading costs
            position_changes = positions['position_size'].diff().fillna(0)
            trading_mask = position_changes != 0
            
            # Entry and exit prices for cost calculation
            positions['entry_price'] = positions['price'].where(trading_mask).ffill()
            positions['position_value'] = positions['position_size'] * positions['price']
            
            # Transaction costs on position value
            transaction_costs = pd.Series(0.0, index=positions.index)
            transaction_costs[trading_mask] = (
                abs(positions['position_value'][trading_mask]) * 
                (costs_config['transaction_fee_pct'] + costs_config['slippage_pct'])
            )
            
            # Borrowing costs for short positions (hourly rate)
            borrowing_costs = pd.Series(0.0, index=positions.index)
            borrowing_costs[positions['position_size'] < 0] = (
                abs(positions['position_value']) * 
                costs_config['borrowing_cost_pa'] / (24 * 252)
            )
            
            # Calculate returns after costs
            market_returns = positions['position_size'] * positions['returns'] * positions['price']
            costs = transaction_costs + borrowing_costs
            net_returns = (market_returns - costs) / notional_value
            
            all_returns.append(net_returns)
        
        # Combine returns across symbols
        if all_returns:
            strategy_returns = pd.concat(all_returns, axis=1).sum(axis=1)  # Sum returns across symbols
        else:
            strategy_returns = pd.Series(0, index=self.signals.index)
            
        return strategy_returns
    
    def update_allocation(self, new_capital: float, prices: Dict[str, float]):
        """
        Update position sizes based on new capital allocation
        
        Returns dictionary of required position adjustments:
        {symbol: (current_size, target_size)}
        """
        if not self.current_positions:
            return {}
            
        position_adjustments = {}
        capital_ratio = new_capital / self.allocated_capital
        
        for symbol, current_size in self.current_positions.items():
            target_size = current_size * capital_ratio
            if target_size != current_size:
                position_adjustments[symbol] = (current_size, target_size)
        
        self.allocated_capital = new_capital
        return position_adjustments

class Portfolio:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.returns = pd.DataFrame()
        self.rebalance_history = []
        
    def add_strategy(self, strategy: Strategy, allocation: float):
        """Add strategy with initial allocation"""
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")
        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        strategy.allocated_capital = self.current_capital * allocation
        
    def rebalance(self, new_allocations: Dict[str, float], prices: Dict[str, float]):
        """
        Rebalance strategy allocations and adjust positions
        
        Parameters:
        -----------
        new_allocations : Dict[str, float]
            New target allocations for each strategy
        prices : Dict[str, float]
            Current prices for all symbols
        """
        if sum(new_allocations.values()) != 1:
            raise ValueError("Allocations must sum to 1")
        
        # Calculate new capital allocations
        position_adjustments = {}
        for strategy_name, new_alloc in new_allocations.items():
            strategy = self.strategies[strategy_name]
            new_capital = self.current_capital * new_alloc
            
            # Get required position adjustments
            strategy_adjustments = strategy.update_allocation(new_capital, prices)
            if strategy_adjustments:
                position_adjustments[strategy_name] = strategy_adjustments
        
        # Record rebalance
        self.rebalance_history.append({
            'timestamp': pd.Timestamp.now(),
            'old_allocations': self.allocations.copy(),
            'new_allocations': new_allocations.copy(),
            'position_adjustments': position_adjustments
        })
        
        # Update allocations
        self.allocations = new_allocations
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict) -> pd.DataFrame:
        """Calculate returns for all strategies and portfolio"""
        strategy_returns = {}
        
        # Calculate returns for each strategy
        for name, strategy in self.strategies.items():
            strategy_returns[name] = strategy.calculate_returns(price_data, costs_config)
        
        # Combine into DataFrame
        self.returns = pd.DataFrame(strategy_returns)
        
        # Calculate portfolio returns with time-varying weights
        portfolio_returns = pd.Series(0.0, index=self.returns.index)
        
        # Apply allocations and rebalancing
        current_allocations = pd.DataFrame(index=self.returns.index, 
                                         columns=self.strategies.keys(),
                                         data=0.0)
        
        # Fill initial allocations
        current_allocations.iloc[0] = pd.Series(self.allocations)
        
        # Apply rebalancing events
        for rebalance in self.rebalance_history:
            timestamp = rebalance['timestamp']
            if timestamp in current_allocations.index:
                current_allocations.loc[timestamp:] = pd.Series(rebalance['new_allocations'])
        
        # Calculate portfolio returns with time-varying weights
        for t in range(len(portfolio_returns)):
            weights = current_allocations.iloc[t]
            portfolio_returns.iloc[t] = (self.returns.iloc[t] * weights).sum()
        
        self.returns['Portfolio'] = portfolio_returns
        return self.returns