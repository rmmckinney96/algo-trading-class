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

class Position:
    def __init__(self, symbol: str, entry_time: pd.Timestamp, entry_price: float, 
                 size: float, direction: int):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.direction = direction
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.costs = 0.0

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
        self.allocated_capital = 0.0
        self.current_positions: Dict[str, Position] = {}
        
        # Pre-allocate trade history
        self.trade_history = pd.DataFrame(
            index=range(1000),  # Adjust size based on expected trades
            columns=['symbol', 'entry_time', 'exit_time', 'entry_price', 
                    'exit_price', 'size', 'direction', 'realized_pnl', 'costs']
        )
        self.trade_count = 0
        
        # Pre-allocate PnL history
        self.pnl_history = pd.DataFrame(
            index=range(len(signals) * len(signals['symbol'].unique())),
            columns=['timestamp', 'symbol', 'realized_pnl', 'unrealized_pnl', 
                    'costs', 'net_pnl', 'capital_used']
        )
        self.pnl_count = 0
    
    def record_trade(self, trade_data: dict):
        """Record trade with pre-allocated DataFrame"""
        if self.trade_count >= len(self.trade_history):
            # Double the size if we need more space
            self.trade_history = pd.concat([
                self.trade_history, 
                pd.DataFrame(index=range(len(self.trade_history)), 
                           columns=self.trade_history.columns)
            ])
        
        for col, value in trade_data.items():
            self.trade_history.iloc[self.trade_count, self.trade_history.columns.get_loc(col)] = value
        self.trade_count += 1
    
    def record_pnl(self, pnl_data: dict):
        """Record PnL with pre-allocated DataFrame"""
        if self.pnl_count >= len(self.pnl_history):
            # Double the size if we need more space
            self.pnl_history = pd.concat([
                self.pnl_history, 
                pd.DataFrame(index=range(len(self.pnl_history)), 
                           columns=self.pnl_history.columns)
            ])
        
        for col, value in pnl_data.items():
            self.pnl_history.iloc[self.pnl_count, self.pnl_history.columns.get_loc(col)] = value
        self.pnl_count += 1
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get completed trade history"""
        return self.trade_history.iloc[:self.trade_count].copy()
    
    def get_pnl_history(self) -> pd.DataFrame:
        """Get PnL history"""
        return self.pnl_history.iloc[:self.pnl_count].copy()
    
    def process_signals(self, price_data: Dict[str, pd.DataFrame], costs_config: Dict):
        """Process signals and manage positions"""
        # Group signals by timestamp
        grouped_signals = self.signals.groupby('symbol')
        
        # Process each symbol's signals
        for symbol, symbol_signals in tqdm(grouped_signals, desc=f"Processing {self.name} signals"):
            if symbol not in price_data:
                continue
                
            prices = price_data[symbol]
            
            # Process each signal
            for timestamp, row in symbol_signals.iterrows():
                if timestamp not in prices.index:
                    continue
                    
                curr_price = prices.loc[timestamp, 'Close']
                signal_value = row['signal']
                
                # Calculate position size based on allocated capital
                position_size = signal_value * self.allocated_capital / curr_price
                
                # Check for existing position
                if symbol in self.current_positions:
                    current_pos = self.current_positions[symbol]
                    
                    if position_size == 0:  # Close position
                        # Calculate exit costs
                        exit_cost = abs(current_pos.size * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        current_pos.costs += exit_cost
                        
                        # Calculate realized PnL
                        price_pnl = (curr_price - current_pos.entry_price) * current_pos.size
                        current_pos.realized_pnl = price_pnl - current_pos.costs
                        
                        # Record trade
                        self.record_trade({
                            'symbol': symbol,
                            'entry_time': current_pos.entry_time,
                            'exit_time': timestamp,
                            'entry_price': current_pos.entry_price,
                            'exit_price': curr_price,
                            'size': current_pos.size,
                            'direction': current_pos.direction,
                            'realized_pnl': current_pos.realized_pnl,
                            'costs': current_pos.costs
                        })
                        
                        # Remove position
                        del self.current_positions[symbol]
                        
                    elif position_size != current_pos.size:  # Adjust position
                        # Calculate adjustment costs
                        size_change = position_size - current_pos.size
                        adjust_cost = abs(size_change * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        current_pos.costs += adjust_cost
                        current_pos.size = position_size
                        
                elif position_size != 0:  # Open new position
                    # Calculate entry costs
                    entry_cost = abs(position_size * curr_price) * (
                        costs_config['transaction_fee_pct'] + 
                        costs_config['slippage_pct']
                    )
                    
                    # Create new position
                    self.current_positions[symbol] = Position(
                        symbol=symbol,
                        entry_time=timestamp,
                        entry_price=curr_price,
                        size=position_size,
                        direction=np.sign(position_size)
                    )
                    self.current_positions[symbol].costs = entry_cost
                
                # Update unrealized PnL for current position
                if symbol in self.current_positions:
                    pos = self.current_positions[symbol]
                    price_pnl = (curr_price - pos.entry_price) * pos.size
                    pos.unrealized_pnl = price_pnl - pos.costs
                    
                    # Add borrowing costs for short positions
                    if pos.size < 0:
                        borrow_cost = (abs(pos.size * curr_price) * 
                                     costs_config['borrowing_cost_pa'] / (24 * 252))
                        pos.costs += borrow_cost
                
                # Record PnL snapshot
                self.record_pnl({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'realized_pnl': pos.realized_pnl if symbol in self.current_positions else 0,
                    'unrealized_pnl': pos.unrealized_pnl if symbol in self.current_positions else 0,
                    'costs': pos.costs if symbol in self.current_positions else 0,
                    'net_pnl': ((pos.realized_pnl + pos.unrealized_pnl) 
                               if symbol in self.current_positions else 0),
                    'capital_used': abs(position_size * curr_price) if position_size != 0 else 0
                })
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get current position summary"""
        summary = []
        for symbol, pos in self.current_positions.items():
            summary.append({
                'symbol': symbol,
                'entry_time': pos.entry_time,
                'entry_price': pos.entry_price,
                'current_size': pos.size,
                'direction': pos.direction,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'costs': pos.costs,
                'net_pnl': pos.realized_pnl + pos.unrealized_pnl
            })
        return pd.DataFrame(summary)

    def update_allocation(self, new_capital: float) -> Dict[str, tuple]:
        """
        Update position sizes based on new capital allocation
        
        Returns dictionary of required position adjustments:
        {symbol: (current_size, target_size)}
        """
        position_adjustments = {}
        
        # Handle case where allocated_capital is zero
        if self.allocated_capital == 0:
            if new_capital > 0:
                # Opening new positions
                for symbol, current_pos in self.current_positions.items():
                    if current_pos.size != 0:
                        position_adjustments[symbol] = (0, current_pos.size)
        else:
            # Normal case - adjust existing positions
            capital_ratio = new_capital / self.allocated_capital
            for symbol, current_pos in self.current_positions.items():
                target_size = current_pos.size * capital_ratio
                if not np.isclose(target_size, current_pos.size):
                    position_adjustments[symbol] = (current_pos.size, target_size)
        
        self.allocated_capital = new_capital
        return position_adjustments

class Portfolio(ABC):
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.allocation_signs: Dict[str, int] = {}  # 1 for normal, -1 for inverse
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
        
        # Initialize returns DataFrame
        self.returns = pd.DataFrame(index=all_timestamps, 
                                  columns=list(self.strategies.keys()) + ['Portfolio'])
        
        # Get rebalancing dates
        dates = pd.date_range(
            start=all_timestamps[0],
            end=all_timestamps[-1],
            freq=rebalance_frequency
        )
        
        # Pre-allocate arrays for faster computation
        n_timestamps = len(all_timestamps)
        portfolio_values = np.zeros(n_timestamps)
        portfolio_values[0] = self.initial_capital
        strategy_returns = {name: np.zeros(n_timestamps) for name in self.strategies}
        
        # Initial allocation and signal processing
        for name, strategy in tqdm(self.strategies.items(), desc="Processing initial signals"):
            strategy.process_signals(price_data, costs_config)
        
        # Process each timestamp
        for i, timestamp in enumerate(tqdm(all_timestamps, desc="Calculating returns")):
            # Check for rebalancing
            if timestamp in dates:
                # Get current prices for position adjustment
                current_prices = {
                    symbol: data.loc[timestamp, 'Close'] 
                    for symbol, data in price_data.items()
                    if timestamp in data.index
                }
                
                # Calculate and apply new allocations
                new_allocations = self.optimize_allocations()
                self.rebalance(new_allocations, current_prices)
            
            # Calculate PnL for this timestamp
            total_pnl = 0
            prev_value = portfolio_values[i-1] if i > 0 else self.initial_capital
            
            # Get PnL for each strategy
            for name, strategy in self.strategies.items():
                position_summary = strategy.get_position_summary()
                if not position_summary.empty:
                    # Calculate net PnL
                    realized = position_summary['realized_pnl'].sum()
                    unrealized = position_summary['unrealized_pnl'].sum()
                    costs = position_summary['costs'].sum()
                    net_pnl = (realized + unrealized - costs) * self.allocation_signs[name]
                    
                    total_pnl += net_pnl
                    strategy_returns[name][i] = net_pnl / prev_value if prev_value != 0 else 0
                else:
                    strategy_returns[name][i] = 0
            
            # Update portfolio value
            portfolio_values[i] = prev_value + total_pnl
            
            # Update PnL tracking
            self.update_pnl_tracking(timestamp)
        
        # Calculate portfolio returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        portfolio_returns = np.insert(portfolio_returns, 0, 0)  # Add 0 return for first timestamp
        
        # Update returns DataFrame efficiently
        for name, returns in strategy_returns.items():
            self.returns[name] = returns
        self.returns['Portfolio'] = portfolio_returns
        
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
        
        # Update PnL tracking after rebalance
        self.update_pnl_tracking(pd.Timestamp.now())

    def update_pnl_tracking(self, timestamp: pd.Timestamp):
        """Update PnL tracking at a given timestamp"""
        # Track strategy-level metrics
        strategy_metrics = []
        for name, strategy in self.strategies.items():
            position_summary = strategy.get_position_summary()
            
            metrics = {
                'timestamp': timestamp,
                'strategy': name,
                'allocated_capital': strategy.allocated_capital,
                'realized_pnl': position_summary['realized_pnl'].sum() if not position_summary.empty else 0,
                'unrealized_pnl': position_summary['unrealized_pnl'].sum() if not position_summary.empty else 0,
                'total_costs': position_summary['costs'].sum() if not position_summary.empty else 0
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
        capital_history = self.capital_history[self.capital_history['strategy'] == strategy_name]
        
        # Get trade history and calculate PnL components
        trades = strategy.get_trade_history()
        if trades.empty:
            return pd.DataFrame()
        
        # Create a DataFrame with all components
        pnl_df = pd.DataFrame({
            'timestamp': capital_history['timestamp'],
            'allocated_capital': capital_history['allocated_capital'],
            'realized_pnl': trades['realized_pnl'].cumsum(),
            'unrealized_pnl': trades['unrealized_pnl'],
            'total_costs': trades['total_costs'].cumsum()
        })
        
        return pnl_df