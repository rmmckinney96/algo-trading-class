import pandas as pd
import numpy as np
from typing import Dict
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

class Positions:
    """Manages a DataFrame of positions and their history"""
    def __init__(self, name: str, initial_capital: float = 0):
        self.name = name
        
        # Initialize positions DataFrame
        self.positions = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'symbol': pd.Series(dtype='str'),
            'strategy': pd.Series(dtype='str'),
            'price': pd.Series(dtype='float64'),
            'size': pd.Series(dtype='float64'),
            'allocated_capital': pd.Series(dtype='float64'),
            'signal': pd.Series(dtype='float64'),
            'returns': pd.Series(dtype='float64'),
            'costs': pd.Series(dtype='float64'),
            'is_active': pd.Series(dtype='bool')
        })
        
        # Initialize cash for all timestamps
        self.cash = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'amount': pd.Series(dtype='float64')
        })
    
    def calculate_costs(self, costs_config: Dict[str, float]):
        """Calculate transaction and borrowing costs"""
        # Calculate absolute size changes
        size_changes = self.positions.groupby('symbol')['size'].diff().abs()
        
        # Calculate transaction costs for each size change
        transaction_costs = (
            size_changes * self.positions['price'] * 
            (costs_config['transaction_fee_pct'] + costs_config['slippage_pct'])
        )
        
        # Calculate borrowing costs for leveraged positions
        leverage = self.positions.groupby('timestamp')['allocated_capital'].transform('sum') / self.cash['amount']
        borrowing_cost = np.maximum(0, leverage - 1) * costs_config['borrowing_cost_pa'] / 252
        
        # Calculate borrowing costs for short positions
        short_costs = (
            (self.positions['size'] < 0) * self.positions['size'].abs() * 
            self.positions['price'] * 
            costs_config['borrowing_cost_pa'] / 252
        )
        
        # Update costs in positions DataFrame
        self.positions['costs'] = (
            transaction_costs + 
            borrowing_cost * self.positions['allocated_capital'] + 
            short_costs
        )
    
    def calculate_returns(self):
        """Calculate position returns and PnL"""
        # Calculate returns for each position
        self.positions['returns'] = (
            self.positions.groupby('symbol')['price'].pct_change() * 
            self.positions.groupby('symbol')['size'].shift()
        )
        
        # Calculate net returns after costs
        self.positions['net_returns'] = (
            self.positions['returns'] - 
            self.positions['costs'] / self.positions['allocated_capital']
        )
        
        # Calculate PnL
        self.positions['pnl'] = (
            self.positions['returns'] * 
            self.positions.groupby('symbol')['allocated_capital'].shift() - 
            self.positions['costs']
        )
        
        # Calculate portfolio value at each timestamp
        self.positions['portfolio_value'] = (
            self.positions.groupby('timestamp')['allocated_capital'].transform('sum') + 
            self.cash['amount']
        )

class Strategy(Positions):
    def __init__(self, name: str, signals: pd.DataFrame, prices: Dict[str, pd.DataFrame]):
        super().__init__(name)
        
        # Create positions DataFrame with all timestamps
        all_positions = []
        
        for symbol, symbol_signals in signals.groupby('symbol'):
            if symbol not in prices:
                continue
            
            price_data = prices[symbol]
            
            # Align signals with prices
            aligned_data = pd.concat([price_data, symbol_signals.set_index('timestamp')], axis=1)
            
            # Forward fill signals
            aligned_data['signal'] = aligned_data['signal'].fillna(method='ffill').fillna(0)
            
            # Create positions for all timestamps
            symbol_positions = pd.DataFrame({
                'timestamp': aligned_data.index,
                'symbol': symbol,
                'strategy': self.name,
                'price': aligned_data['Close'],
                'signal': aligned_data['signal'],
                'size': 0.0,
                'allocated_capital': 0.0,
                'returns': 0.0,
                'costs': 0.0,
                'is_active': False
            })
            
            all_positions.append(symbol_positions)
        
        # Combine all positions and sort by timestamp
        if all_positions:
            self.positions = pd.concat(all_positions, ignore_index=True)
            self.positions = self.positions.sort_values('timestamp')
        
        # Initialize cash for all timestamps
        self.cash = pd.DataFrame({
            'timestamp': self.positions['timestamp'].unique(),
            'amount': 0.0
        }).set_index('timestamp')

class Portfolio(Positions, ABC):
    def __init__(self, initial_capital: float = 1000000):
        super().__init__("Portfolio", initial_capital)
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.allocation_signs: Dict[str, int] = {}
    
    def add_strategy(self, strategy: Strategy, allocation: float, inverse: bool = False):
        """Add strategy with initial allocation"""
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")
        
        # Calculate strategy capital
        strategy_capital = self.cash['amount'].iloc[0] * allocation
        
        # Update strategy cash
        strategy.cash['amount'] = strategy_capital
        
        # Store strategy info
        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        self.allocation_signs[strategy.name] = -1 if inverse else 1
    
    @abstractmethod
    def optimize_allocations(self) -> Dict[str, float]:
        """Calculate optimal allocations for strategies"""
        pass
    
    def calculate_returns(self, costs_config: Dict) -> pd.DataFrame:
        """Calculate portfolio returns with periodic rebalancing"""
        # Get all timestamps from strategies
        all_timestamps = pd.Series(dtype='datetime64[ns]')
        for strategy in self.strategies.values():
            all_timestamps = all_timestamps.union(strategy.positions['timestamp'].unique())
        all_timestamps = all_timestamps.sort_values()
        
        # Initialize portfolio positions
        all_positions = []
        for strategy in self.strategies.values():
            all_positions.append(strategy.positions)
        self.positions = pd.concat(all_positions, ignore_index=True)
        
        # Process each timestamp
        for timestamp in all_timestamps:
            # Calculate returns up to this point
            super().calculate_returns()  # Call parent class method
            
            # Get new allocations
            new_allocations = self.optimize_allocations()
            
            # Update strategy allocations and position sizes
            total_value = self.positions[
                self.positions['timestamp'] == timestamp
            ]['portfolio_value'].iloc[0]
            
            for name, allocation in new_allocations.items():
                strategy = self.strategies[name]
                strategy_capital = total_value * allocation
                
                # Update strategy positions from this timestamp forward
                future_mask = strategy.positions['timestamp'] >= timestamp
                current_positions = strategy.positions[future_mask]
                
                # Scale position sizes by new capital ratio
                if not current_positions.empty:
                    old_capital = current_positions['allocated_capital'].sum()
                    if old_capital > 0:
                        capital_ratio = strategy_capital / old_capital
                        strategy.positions.loc[future_mask, 'size'] *= capital_ratio
                        strategy.positions.loc[future_mask, 'allocated_capital'] *= capital_ratio
                
                # Update strategy cash
                strategy.cash.loc[timestamp:, 'amount'] = strategy_capital
            
            # Update portfolio positions
            all_positions = []
            for strategy in self.strategies.values():
                all_positions.append(strategy.positions)
            self.positions = pd.concat(all_positions, ignore_index=True)
            
            # Calculate costs
            self.calculate_costs(costs_config)
        
        return self.positions.groupby('timestamp')[['returns', 'pnl']].sum()


class EqualWeightPortfolio(Portfolio):
    """Simple portfolio that allocates capital equally among all strategies"""
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Equal weight allocation to all strategies"""
        if not self.strategies:
            return {}
        
        allocation = 1.0 / len(self.strategies)
        return {name: allocation for name in self.strategies}


class KellyPortfolio(Portfolio):
    def __init__(self, initial_capital: float = 1000000, lookback_days: int = 30):
        super().__init__(initial_capital)
        self.lookback_hours = lookback_days * 24
    
    def calculate_kelly_score(self, strategy_name: str) -> float:
        """Calculate Kelly score for a strategy using net returns"""
        strategy = self.strategies[strategy_name]
        
        # Get recent positions data
        recent_positions = strategy.positions.tail(self.lookback_hours)
        if len(recent_positions) < self.lookback_hours:
            return 0.0
        
        # Use net returns that already include all costs
        net_returns = recent_positions['net_returns']
        
        wins = net_returns > 0
        losses = net_returns < 0
        
        # If no wins and no losses, allocate equally
        if len(net_returns[wins]) == 0 and len(net_returns[losses]) == 0:
            return 1.0 / len(self.strategies)
        
        win_prob = len(net_returns[wins]) / len(net_returns)
        avg_win = net_returns[wins].mean()
        avg_loss = abs(net_returns[losses].mean())
        
        if avg_loss == 0:  # Avoid division by zero
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        
        return kelly  # Allow negative scores for inverse positions
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Calculate optimal allocations using Kelly Criterion"""
        if not self.strategies:
            return {}
        
        # Calculate Kelly scores for each strategy
        kelly_scores = {
            name: self.calculate_kelly_score(name)
            for name in self.strategies
        }
        
        # Use absolute values for normalization but keep signs for direction
        abs_total = sum(abs(score) for score in kelly_scores.values())
        if abs_total == 0:
            # If all scores are 0, use equal weights
            allocation = 1.0 / len(self.strategies)
            return {name: allocation for name in self.strategies}
        
        # Calculate allocations preserving signs
        allocations = {
            name: score / abs_total 
            for name, score in kelly_scores.items()
        }
        
        # Print Kelly scores and allocations
        print("\nKelly Portfolio Allocations:")
        for name, alloc in allocations.items():
            leverage = abs(alloc * abs_total)
            direction = "Short" if alloc < 0 else "Long"
            print(f"{name}: {alloc:.1%} ({direction}, Kelly: {kelly_scores[name]:.2f}, Leverage: {leverage:.2f}x)")
        
        return allocations
