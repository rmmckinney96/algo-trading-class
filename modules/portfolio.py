import pandas as pd
import numpy as np
from typing import Dict, Optional
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os
import concurrent.futures
import copy

class Position:
    """Represents a trading position with its full lifecycle and risk metrics."""
    def __init__(self, 
                 symbol: str, 
                 entry_time: pd.Timestamp, 
                 entry_price: float,
                 size: float,
                 direction: int,
                 strategy_name: str = None):
        # Core position data
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size  # Positive for long, negative for short
        self.direction = direction  # 1 for long, -1 for short
        self.strategy_name = strategy_name
        
        # Position status
        self.is_open = True
        self.exit_time = None
        self.exit_price = None
        
        # PnL components
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.costs = 0.0
        
        # Risk metrics
        self.max_favorable_excursion = 0.0  # Maximum profit reached
        self.max_adverse_excursion = 0.0    # Maximum drawdown reached
        self.high_water_mark = entry_price
        self.low_water_mark = entry_price
        
        # Trade history
        self.price_history = []
        self.size_adjustments = []
    
    def update(self, current_price: float, timestamp: pd.Timestamp):
        """Update position with current market price."""
        if not self.is_open:
            return
        
        # Update price history
        self.price_history.append((timestamp, current_price))
        
        # Calculate unrealized PnL
        price_change = current_price - self.entry_price
        self.unrealized_pnl = price_change * self.size - self.costs
        
        # Update water marks and excursions
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
            favorable_excursion = (self.high_water_mark - self.entry_price) * self.size
            self.max_favorable_excursion = max(self.max_favorable_excursion, favorable_excursion)
        
        if current_price < self.low_water_mark:
            self.low_water_mark = current_price
            adverse_excursion = (self.entry_price - self.low_water_mark) * self.size
            self.max_adverse_excursion = max(self.max_adverse_excursion, adverse_excursion)
    
    def adjust_size(self, 
                   new_size: float, 
                   price: float, 
                   timestamp: pd.Timestamp, 
                   costs: float = 0.0):
        """Adjust position size and record the change."""
        if not self.is_open:
            return
        
        size_change = new_size - self.size
        self.size_adjustments.append({
            'timestamp': timestamp,
            'old_size': self.size,
            'new_size': new_size,
            'price': price,
            'costs': costs
        })
        
        self.size = new_size
        self.costs += costs
        self.update(price, timestamp)
    
    def close(self, exit_price: float, exit_time: pd.Timestamp, costs: float = 0.0):
        """Close the position and realize PnL."""
        if not self.is_open:
            return
        
        self.is_open = False
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.costs += costs
        
        # Calculate final realized PnL
        price_change = exit_price - self.entry_price
        self.realized_pnl = price_change * self.size - self.costs
        self.unrealized_pnl = 0.0
    
    def get_metrics(self) -> dict:
        """Get current position metrics."""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'current_size': self.size,
            'direction': self.direction,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'costs': self.costs,
            'net_pnl': self.realized_pnl + self.unrealized_pnl - self.costs,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'is_open': self.is_open,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price
        }

class Positions:
    """Base class for managing positions and tracking their history"""
    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, Position] = {}  # Current open positions
        self.closed_positions = []  # History of closed positions
        
        # History tracking
        self.capital_history = pd.DataFrame(columns=[
            'timestamp', 'allocated_capital', 
            'realized_pnl', 'unrealized_pnl', 'total_costs'
        ])
        self.pnl_history = pd.DataFrame(columns=[
            'timestamp', 'portfolio_value', 'total_allocated',
            'total_realized_pnl', 'unrealized_pnl', 'total_costs'
        ])
        self.trade_history = pd.DataFrame(columns=[
            'symbol', 'entry_time', 'exit_time', 'entry_price', 
            'exit_price', 'size', 'direction', 'realized_pnl',
            'costs', 'max_favorable_excursion', 'max_adverse_excursion'
        ])
    
    def get_position_summary(self, timestamp: pd.Timestamp = None) -> pd.DataFrame:
        """Get summary of positions up to a specific timestamp"""
        if not self.positions and not self.closed_positions:
            return pd.DataFrame()
        
        position_metrics = []
        processed_positions = set()  # Track which positions we've already processed
        
        # Add open positions
        for position in self.positions.values():
            if timestamp and position.entry_time > timestamp:
                continue
            
            metrics = position.get_metrics()
            if timestamp:
                relevant_prices = [(t, p) for t, p in position.price_history if t <= timestamp]
                if relevant_prices:
                    last_price = relevant_prices[-1][1]
                    position.update(last_price, timestamp)
                    metrics = position.get_metrics()
            
            position_metrics.append(metrics)
            processed_positions.add(position.symbol)  # Track that we've processed this position
        
        # Add closed positions
        for position in self.closed_positions:
            # Skip if we've already processed this position
            if position.symbol in processed_positions:
                continue
            
            if timestamp and position.exit_time > timestamp:
                # Position was still open at timestamp
                relevant_prices = [(t, p) for t, p in position.price_history if t <= timestamp]
                if relevant_prices:
                    last_price = relevant_prices[-1][1]
                    position_copy = copy.deepcopy(position)
                    position_copy.is_open = True
                    position_copy.exit_time = None
                    position_copy.exit_price = None
                    position_copy.update(last_price, timestamp)
                    metrics = position_copy.get_metrics()
                    position_metrics.append(metrics)
                    processed_positions.add(position.symbol)
            elif not timestamp or position.exit_time <= timestamp:
                # Position was already closed
                metrics = position.get_metrics()
                position_metrics.append(metrics)
                processed_positions.add(position.symbol)
        
        if not position_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(position_metrics)
    
    def record_trade(self, position: Position):
        """Record a closed trade"""
        if position.exit_time is not None:
            trade_data = {
                'symbol': position.symbol,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'size': position.size,
                'direction': position.direction,
                'realized_pnl': position.realized_pnl,
                'costs': position.costs,
                'max_favorable_excursion': position.max_favorable_excursion,
                'max_adverse_excursion': position.max_adverse_excursion
            }
            self.trade_history = pd.concat([
                self.trade_history,
                pd.DataFrame([trade_data])
            ], ignore_index=True)
    
    def update_pnl(self, timestamp: pd.Timestamp, allocated_capital: float = None, 
                   allocation_sign: int = 1) -> Dict[str, float]:
        """
        Update PnL tracking at given timestamp
        
        Parameters:
        -----------
        timestamp : pd.Timestamp
            Current timestamp
        allocated_capital : float, optional
            Current allocated capital
        allocation_sign : int, optional
            1 for normal positions, -1 for inverse positions
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing PnL metrics and returns
        """
        position_summary = self.get_position_summary(timestamp)
        prev_value = self.pnl_history['portfolio_value'].iloc[-1] if not self.pnl_history.empty else allocated_capital
        
        if position_summary.empty:
            metrics = {
                'timestamp': timestamp,
                'allocated_capital': allocated_capital if allocated_capital is not None else 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'total_costs': 0
            }
            strategy_return = 0
        else:
            realized = position_summary['realized_pnl'].sum()
            unrealized = position_summary['unrealized_pnl'].sum()
            costs = position_summary['costs'].sum()
            
            # Apply allocation sign
            net_pnl = (realized + unrealized - costs) * allocation_sign
            
            metrics = {
                'timestamp': timestamp,
                'allocated_capital': allocated_capital if allocated_capital is not None else 0,
                'realized_pnl': realized * allocation_sign,
                'unrealized_pnl': unrealized * allocation_sign,
                'total_costs': costs
            }
            
            # Calculate return
            strategy_return = net_pnl / prev_value if prev_value != 0 else 0
        
        # Update capital history
        self.capital_history = pd.concat([
            self.capital_history,
            pd.DataFrame([metrics])
        ], ignore_index=True)
        
        # Calculate portfolio value
        portfolio_value = (metrics['allocated_capital'] + 
                         metrics['realized_pnl'] + 
                         metrics['unrealized_pnl'] - 
                         metrics['total_costs'])
        
        # Update PnL history
        pnl_metrics = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'total_allocated': metrics['allocated_capital'],
            'total_realized_pnl': metrics['realized_pnl'],
            'unrealized_pnl': metrics['unrealized_pnl'],
            'total_costs': metrics['total_costs']
        }
        
        self.pnl_history = pd.concat([
            self.pnl_history,
            pd.DataFrame([pnl_metrics])
        ], ignore_index=True)
        
        return {
            'metrics': metrics,
            'portfolio_value': portfolio_value,
            'return': strategy_return
        }

class Strategy(Positions):
    def __init__(self, name: str, signals: pd.DataFrame):
        super().__init__(name)
        self.signals = signals.set_index('timestamp').copy()
        self.allocated_capital = 0.0
    
    def process_signals(self, price_data: Dict[str, pd.DataFrame], costs_config: Dict, 
                       start_time: pd.Timestamp = None, portfolio: 'Portfolio' = None):
        """Process signals and manage positions"""
        # Group signals by symbol
        grouped_signals = self.signals.groupby('symbol')
        
        # Process each symbol's signals
        for symbol, symbol_signals in grouped_signals:
            if symbol not in price_data:
                continue
                
            prices = price_data[symbol]
            
            # Filter signals based on start_time if provided
            if start_time is not None:
                symbol_signals = symbol_signals[symbol_signals.index >= start_time]
            
            # Process each signal
            for timestamp, row in symbol_signals.iterrows():
                if timestamp not in prices.index:
                    continue
                    
                # Skip if we have no capital allocated
                if self.allocated_capital == 0:
                    continue
                    
                curr_price = prices.loc[timestamp, 'Close']
                signal_value = row['signal']
                
                # Calculate position size based on allocated capital
                target_size = signal_value * self.allocated_capital / curr_price
                
                # Update or close existing position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Update position with current price
                    position.update(curr_price, timestamp)
                    
                    if target_size == 0:  # Close position
                        # Calculate exit costs
                        exit_cost = abs(position.size * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        
                        # Close position
                        position.close(curr_price, timestamp, exit_cost)
                        self.closed_positions.append(position)
                        
                        # Notify portfolio of closed position
                        if portfolio is not None:
                            portfolio.record_closed_position(self.name, position)
                        
                        del self.positions[symbol]
                    
                    elif target_size != position.size:  # Adjust position
                        # Calculate adjustment costs
                        size_change = target_size - position.size
                        adjust_cost = abs(size_change * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        
                        # Adjust position size
                        position.adjust_size(target_size, curr_price, timestamp, adjust_cost)
                
                elif target_size != 0:  # Open new position
                    # Calculate entry costs
                    entry_cost = abs(target_size * curr_price) * (
                        costs_config['transaction_fee_pct'] + 
                        costs_config['slippage_pct']
                    )
                    
                    # Create new position
                    position = Position(
                        symbol=symbol,
                        entry_time=timestamp,
                        entry_price=curr_price,
                        size=target_size,
                        direction=np.sign(target_size),
                        strategy_name=self.name
                    )
                    position.costs = entry_cost
                    self.positions[symbol] = position
    
    def update_allocation(self, new_capital: float) -> Dict[str, tuple]:
        """
        Update position sizes based on new capital allocation
        
        Parameters:
        -----------
        new_capital : float
            New capital allocation for this strategy
            
        Returns:
        --------
        Dict[str, tuple]
            Dictionary mapping symbols to (current_size, target_size) tuples
        """
        position_adjustments = {}
        
        # Skip if no change in allocation
        if np.isclose(new_capital, self.allocated_capital):
            return position_adjustments
        
        # Calculate the ratio of new to old capital
        if self.allocated_capital > 0:
            capital_ratio = new_capital / self.allocated_capital
            
            # Adjust each position's size proportionally
            for symbol, position in self.positions.items():
                if position.is_open:
                    target_size = position.size * capital_ratio
                    if not np.isclose(target_size, position.size):
                        position_adjustments[symbol] = (position.size, target_size)
        
        # Update allocated capital
        old_capital = self.allocated_capital
        self.allocated_capital = new_capital
        
        return position_adjustments
    

class Portfolio(Positions, ABC):
    def __init__(self, initial_capital: float = 1000000):
        super().__init__("Portfolio")
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.allocation_signs: Dict[str, int] = {}
        self.cash_allocation = 0.0
        self.rebalance_history = []
        
        # Track capital and PnL
        self.capital_history = pd.DataFrame(columns=[
            'timestamp', 'strategy', 'allocated_capital', 
            'realized_pnl', 'unrealized_pnl', 'total_costs'
        ])
        self.total_pnl = pd.DataFrame(columns=[
            'timestamp', 'total_allocated', 'total_realized_pnl',
            'total_unrealized_pnl', 'total_costs', 'cash_balance',
            'portfolio_value'
        ])
        self.returns = pd.DataFrame()
    
    def _update_strategy_allocation(self, strategy_name: str, new_alloc: float) -> tuple:
        """Update a single strategy's allocation"""
        strategy = self.strategies[strategy_name]
        new_capital = self.current_capital * new_alloc
        strategy_adjustments = strategy.update_allocation(new_capital)
        return strategy_name, strategy_adjustments
    
    def add_strategy(self, strategy: Strategy, allocation: float, inverse: bool = False):
        """
        Add strategy with initial allocation
        
        Parameters:
        -----------
        strategy : Strategy
            Strategy to add
        allocation : float
            Capital allocation (0 to 1)
        inverse : bool
            Whether to take inverse positions
        """
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")
            
        # Update cash allocation
        self.cash_allocation = max(0, 1 - (sum(self.allocations.values()) + allocation))
        
        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        self.allocation_signs[strategy.name] = -1 if inverse else 1
        strategy.allocated_capital = self.current_capital * allocation
    
    @abstractmethod
    def optimize_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal allocations for each strategy
        
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping strategy names to their allocations (0 to 1)
            The sum of allocations should be <= 1, with remainder going to cash
        """
        pass
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict,
                         rebalance_frequency: str = '30D') -> pd.DataFrame:
        """Calculate portfolio returns and rebalance periodically"""
        # Get all timestamps from price data
        all_timestamps = sorted(set().union(*[df.index for df in price_data.values()]))
        
        # Get rebalancing dates
        rebalance_dates = pd.date_range(
            start=all_timestamps[0],
            end=all_timestamps[-1],
            freq=rebalance_frequency
        )
        
        # Initialize returns DataFrame
        self.returns = pd.DataFrame(index=all_timestamps, 
                                  columns=list(self.strategies.keys()) + ['Portfolio'])
        
        # Initial signal processing with allocated capital
        for name, strategy in self.strategies.items():
            strategy.process_signals(price_data, costs_config, 
                                   start_time=all_timestamps[0],
                                   portfolio=self)
        
        # Process each timestamp
        for i, timestamp in enumerate(tqdm(all_timestamps, desc="Calculating returns")):
            # Check for rebalancing
            if timestamp in rebalance_dates:
                # Get current prices for position adjustment
                current_prices = {
                    symbol: data.loc[timestamp, 'Close'] 
                    for symbol, data in price_data.items()
                    if timestamp in data.index
                }
                
                # Calculate and apply new allocations
                new_allocations = self.optimize_allocations()
                self.rebalance(new_allocations, current_prices)
                
                # Process signals after rebalancing
                for name, strategy in self.strategies.items():
                    strategy.process_signals(price_data, costs_config, 
                                          start_time=timestamp,
                                          portfolio=self)
            
            # Update PnL for each strategy
            strategy_returns = {}
            total_pnl = 0
            for name, strategy in self.strategies.items():
                pnl_data = strategy.update_pnl(
                    timestamp, 
                    allocated_capital=strategy.allocated_capital,
                    allocation_sign=self.allocation_signs[name]
                )
                strategy_returns[name] = pnl_data['return']
                total_pnl += pnl_data['portfolio_value']
            
            # Update portfolio PnL
            portfolio_data = self.update_pnl(timestamp, self.current_capital)
            
            # Record returns
            for strategy_name, strategy_return in strategy_returns.items():
                self.returns.loc[timestamp, strategy_name] = strategy_return
            
            if i > 0:
                portfolio_return = portfolio_data['return']
                self.returns.loc[timestamp, 'Portfolio'] = portfolio_return
        
        return self.returns

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
        # Verify allocations including cash
        total_allocation = sum(new_allocations.values()) + self.cash_allocation
        if not np.isclose(total_allocation, 1.0, rtol=1e-5):
            raise ValueError(f"Allocations (including cash) must sum to 1, got {total_allocation}")
        
        # Update each strategy's allocation
        position_adjustments = {}
        for strategy_name, new_alloc in new_allocations.items():
            name, adjustments = self._update_strategy_allocation(strategy_name, new_alloc)
            if adjustments:
                position_adjustments[name] = adjustments
                
                # Check if any positions were closed during rebalancing
                strategy = self.strategies[name]
                for position in strategy.closed_positions:
                    if position not in self.closed_positions:
                        self.record_closed_position(name, position)
        
        # Record rebalance
        self.rebalance_history.append({
            'timestamp': pd.Timestamp.now(),
            'old_allocations': self.allocations.copy(),
            'new_allocations': new_allocations.copy(),
            'cash_allocation': self.cash_allocation,
            'position_adjustments': position_adjustments
        })
        
        # Update allocations
        self.allocations = new_allocations
        

    def update_pnl_tracking(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Update PnL tracking at a given timestamp"""
        # Track strategy-level metrics
        strategy_metrics = []
        strategy_returns = {}
        total_pnl = 0
        prev_value = self.total_pnl['portfolio_value'].iloc[-1] if not self.total_pnl.empty else self.initial_capital
        
        for name, strategy in self.strategies.items():
            # Get position summary up to this timestamp
            position_summary = strategy.get_position_summary(timestamp)
            
            if not position_summary.empty:
                # Calculate PnL components
                realized = position_summary['realized_pnl'].sum()
                unrealized = position_summary['unrealized_pnl'].sum()
                costs = position_summary['costs'].sum()
                
                # Apply allocation sign (for inverse positions)
                net_pnl = (realized + unrealized - costs) * self.allocation_signs[name]
                
                # Calculate return for this strategy
                strategy_returns[name] = net_pnl / prev_value if prev_value != 0 else 0
                
                # Add to total PnL
                total_pnl += net_pnl
                
                # Record metrics
                metrics = {
                    'timestamp': timestamp,
                    'strategy': name,
                    'allocated_capital': strategy.allocated_capital,
                    'realized_pnl': realized,
                    'unrealized_pnl': unrealized,
                    'total_costs': costs
                }
            else:
                strategy_returns[name] = 0
                metrics = {
                    'timestamp': timestamp,
                    'strategy': name,
                    'allocated_capital': strategy.allocated_capital,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'total_costs': 0
                }
            
            strategy_metrics.append(metrics)
        
        # Update capital history
        new_capital_history = pd.DataFrame(strategy_metrics)
        self.capital_history = pd.concat([self.capital_history, new_capital_history], ignore_index=True)
        
        # Calculate portfolio totals
        total_allocated = sum(s.allocated_capital for s in self.strategies.values())
        total_realized = sum(m['realized_pnl'] for m in strategy_metrics)
        total_unrealized = sum(m['unrealized_pnl'] for m in strategy_metrics)
        total_costs = sum(m['total_costs'] for m in strategy_metrics)
        cash_balance = self.current_capital * self.cash_allocation
        portfolio_value = (total_allocated + total_realized + total_unrealized - 
                          total_costs + cash_balance)
        
        # Update total PnL tracking
        self.total_pnl = pd.concat([
            self.total_pnl,
            pd.DataFrame([{
                'timestamp': timestamp,
                'total_allocated': total_allocated,
                'total_realized_pnl': total_realized,
                'total_unrealized_pnl': total_unrealized,
                'total_costs': total_costs,
                'cash_balance': cash_balance,
                'portfolio_value': portfolio_value
            }])
        ], ignore_index=True)
        
        return {
            'strategy_returns': strategy_returns,
            'total_pnl': total_pnl,
            'portfolio_value': portfolio_value
        }

    def get_portfolio_positions(self) -> pd.DataFrame:
        """Get current positions from all strategies in the portfolio.
        
        Returns:
            DataFrame with columns: strategy, symbol, entry_time, entry_price, 
            current_size, direction, unrealized_pnl, realized_pnl, costs, net_pnl
        """
        positions = []
        
        for strategy_name, strategy in self.strategies.items():
            # Get position summary for this strategy
            strategy_positions = strategy.get_position_summary()
            
            if not strategy_positions.empty:
                # Add strategy name to positions
                strategy_positions['strategy'] = strategy_name
                positions.append(strategy_positions)
        
        if positions:
            # Combine all positions
            all_positions = pd.concat(positions, ignore_index=True)
            
            # Reorder columns for better readability
            cols = ['strategy', 'symbol', 'entry_time', 'entry_price', 'current_size',
                    'direction', 'unrealized_pnl', 'realized_pnl', 'costs', 'net_pnl']
            all_positions = all_positions[cols]
            
            return all_positions
        
        return pd.DataFrame(columns=['strategy', 'symbol', 'entry_time', 'entry_price', 
                                   'current_size', 'direction', 'unrealized_pnl', 
                                   'realized_pnl', 'costs', 'net_pnl'])

    def get_pnl_summary(self) -> pd.DataFrame:
        """Generate a summary of portfolio PnL metrics."""
        if self.total_pnl is None:
            return pd.DataFrame()
        
        returns = self.total_pnl.set_index('timestamp')['portfolio_value'].pct_change()
        cumulative_returns = (1 + returns).cumprod()
        drawdowns = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
        
        summary = pd.DataFrame({
            'Metric': [
                'Initial Capital',
                'Current Portfolio Value',
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Current Drawdown'
            ],
            'Value': [
                f"${self.initial_capital:,.2f}",
                f"${self.total_pnl['portfolio_value'].iloc[-1]:,.2f}",
                f"{(cumulative_returns.iloc[-1] - 1):.2%}",
                f"{(((1 + (cumulative_returns.iloc[-1] - 1)) ** (252/len(returns))) - 1):.2%}",
                f"{returns.std() * np.sqrt(252):.2%}",
                f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}",
                f"{drawdowns.min():.2%}",
                f"{drawdowns.iloc[-1]:.2%}"
            ]
        })
        
        return summary
    
    def get_strategy_pnl(self, strategy_name: str) -> pd.DataFrame:
        """Get PnL breakdown for a specific strategy."""
        if strategy_name not in self.strategies:
            return pd.DataFrame()
        
        strategy = self.strategies[strategy_name]
        
        # Get capital allocation history for this strategy
        capital_history = self.capital_history[self.capital_history['strategy'] == strategy_name].copy()
        if capital_history.empty:
            return pd.DataFrame()
        
        # Get the first timestamp where capital was allocated
        allocation_start = capital_history['timestamp'].min()
        
        # Get trade history and calculate PnL components
        trades = strategy.get_trade_history()
        if not trades.empty:
            # Filter trades to only include those after allocation
            trades = trades[trades['entry_time'] >= allocation_start].copy()
        
        if trades.empty:
            return pd.DataFrame()
        
        # Get PnL history for unrealized PnL
        pnl_history = strategy.get_pnl_history()
        if not pnl_history.empty:
            # Filter PnL history to only include after allocation
            pnl_history = pnl_history[pnl_history['timestamp'] >= allocation_start].copy()
        
        # Create time series for realized PnL and costs
        trade_ts = pd.DataFrame(index=capital_history['timestamp'])
        trade_ts['realized_pnl'] = 0.0
        trade_ts['costs'] = 0.0
        
        # Fill in realized PnL and costs at trade timestamps
        for _, trade in trades.iterrows():
            if trade['exit_time'] in trade_ts.index:
                trade_ts.loc[trade['exit_time'], 'realized_pnl'] = trade['realized_pnl']
                trade_ts.loc[trade['exit_time'], 'costs'] = trade['costs']
        
        # Cumulative sum of realized PnL and costs
        trade_ts['realized_pnl'] = trade_ts['realized_pnl'].cumsum()
        trade_ts['costs'] = trade_ts['costs'].cumsum()
        
        # Get unrealized PnL from PnL history
        if not pnl_history.empty:
            pnl_history = pnl_history.set_index('timestamp')
            trade_ts['unrealized_pnl'] = pnl_history['unrealized_pnl']
        else:
            trade_ts['unrealized_pnl'] = 0.0
        
        # Create final DataFrame
        pnl_df = pd.DataFrame({
            'timestamp': capital_history['timestamp'],
            'allocated_capital': capital_history['allocated_capital'],
            'realized_pnl': trade_ts['realized_pnl'],
            'unrealized_pnl': trade_ts['unrealized_pnl'],
            'total_costs': trade_ts['costs']
        })
        
        # Forward fill any NaN values
        pnl_df = pnl_df.fillna(method='ffill')
        # Fill any remaining NaN values with 0
        pnl_df = pnl_df.fillna(0)
        
        return pnl_df

    def record_closed_position(self, strategy_name: str, position: Position):
        """Record a closed position from a strategy at the portfolio level"""
        # Create a copy of the position with portfolio-level adjustments
        portfolio_position = copy.deepcopy(position)
        portfolio_position.strategy_name = strategy_name
        
        # Apply allocation sign to PnL components
        allocation_sign = self.allocation_signs.get(strategy_name, 1)
        portfolio_position.realized_pnl *= allocation_sign
        portfolio_position.unrealized_pnl *= allocation_sign
        
        # Add to closed positions
        self.closed_positions.append(portfolio_position)
        
        # Record trade in trade history
        self.record_trade(portfolio_position)







class KellyPortfolio(Portfolio):
    def __init__(self, initial_capital: float = 1000000, lookback_days: int = 30):
        super().__init__(initial_capital)
        self.lookback_hours = lookback_days * 24
        self.bad_strategy_threshold = -0.5  # Kelly score below this is considered "bad"
    
    def calculate_kelly_criterion(self, pnl_history: pd.DataFrame) -> float:
        """Calculate Kelly Criterion for a strategy's PnL history"""
        if len(pnl_history) < self.lookback_hours:
            return 0.0
            
        recent_pnl = pnl_history.iloc[-self.lookback_hours:]
        
        # Calculate PnL changes
        pnl_changes = recent_pnl['net_pnl'].diff().dropna()
        
        wins = pnl_changes > 0
        losses = pnl_changes < 0
        
        if len(pnl_changes[losses]) == 0:  # No losses
            return 1.0
        if len(pnl_changes[wins]) == 0:  # No wins
            return -1.0
            
        win_prob = len(pnl_changes[wins]) / len(pnl_changes)
        avg_win = pnl_changes[wins].mean() if len(pnl_changes[wins]) > 0 else 0
        avg_loss = abs(pnl_changes[losses].mean())
        
        if avg_loss == 0:  # Avoid division by zero
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        return kelly  # Return raw score (can be negative)
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Calculate optimal allocations using Kelly Criterion"""
        kelly_scores = {}
        
        # Calculate Kelly Criterion for each strategy
        for name, strategy in self.strategies.items():
            pnl_history = strategy.get_pnl_history()
            kelly = self.calculate_kelly_criterion(pnl_history)
            kelly_scores[name] = kelly
            
            # Update allocation signs based on Kelly scores
            if kelly < self.bad_strategy_threshold:
                # Strategy is bad enough to inverse
                self.allocation_signs[name] = -1
                kelly_scores[name] = abs(kelly)  # Use absolute value for allocation
            else:
                self.allocation_signs[name] = 1
        
        # Filter for strategies with significant Kelly scores
        good_strategies = {k: v for k, v in kelly_scores.items() 
                         if abs(v) > 0.1}  # Use absolute value for filtering
        
        if good_strategies:
            # Normalize scores to sum to 1
            total_score = sum(abs(v) for v in good_strategies.values())
            allocations = {k: abs(v)/total_score for k, v in good_strategies.items()}
            
            # Add zero allocations for strategies not selected
            for strategy in self.strategies:
                if strategy not in allocations:
                    allocations[strategy] = 0.0
            
            self.cash_allocation = 0.0
            
            # Print allocation decisions
            print("\nKelly Portfolio Allocations:")
            for strategy, alloc in allocations.items():
                if alloc > 0:
                    direction = "INVERSE" if self.allocation_signs[strategy] < 0 else "NORMAL"
                    print(f"{strategy}: {alloc:.1%} ({direction}) - Kelly: {kelly_scores[strategy]:.2f}")
        else:
            # If no strategies have significant Kelly scores, go to cash
            allocations = {k: 0.0 for k in self.strategies}
            self.cash_allocation = 1.0
            print("\nNo strategies meet Kelly criterion - Going to cash")
        
        return allocations

class EqualWeightPortfolio(Portfolio):
    """Simple portfolio that allocates capital equally among all strategies"""
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Equal weight allocation to all strategies"""
        n_strategies = len(self.strategies)
        if n_strategies == 0:
            return {}
            
        allocation = 1.0 / n_strategies
        allocations = {name: allocation for name in self.strategies}
        self.cash_allocation = 0.0
        
        return allocations
