import pandas as pd
import numpy as np
from typing import Dict, Optional
from tqdm.auto import tqdm

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
        
        # Pre-allocate trade history with a reasonable size
        self.trade_history = pd.DataFrame(
            index=range(1000),  # Adjust size based on expected trades
            columns=['symbol', 'entry_time', 'exit_time', 'entry_price', 
                    'exit_price', 'size', 'direction', 'realized_pnl', 'costs']
        )
        self.trade_count = 0
        
        # Pre-allocate PnL history with index matching signals
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
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict, 
                         allocation_sign: int = 1) -> pd.Series:
        """Calculate returns and track positions/PnL"""
        all_returns = []
        
        for symbol in self.signals['symbol'].unique():
            if symbol not in price_data:
                continue
            
            prices = price_data[symbol]
            symbol_signals = self.signals[self.signals['symbol'] == symbol]['signal']
            
            # Initialize position tracking for this symbol
            positions = pd.DataFrame(index=prices.index)
            positions['signal'] = symbol_signals.reindex(prices.index).ffill().fillna(0) * allocation_sign
            positions['price'] = prices['Close']
            positions['size'] = positions['signal'] * self.allocated_capital / positions['price']
            
            # Track PnL for each timestamp
            pnl_entries = []
            current_position = None
            
            for t in range(len(positions)):
                timestamp = positions.index[t]
                curr_price = positions['price'].iloc[t]
                curr_size = positions['size'].iloc[t]
                
                # Check for position changes
                if current_position is None and curr_size != 0:
                    # Opening new position
                    current_position = Position(
                        symbol=symbol,
                        entry_time=timestamp,
                        entry_price=curr_price,
                        size=curr_size,
                        direction=np.sign(curr_size)
                    )
                    self.current_positions[symbol] = current_position
                    
                    # Calculate entry costs
                    entry_cost = abs(curr_size * curr_price) * (
                        costs_config['transaction_fee_pct'] + 
                        costs_config['slippage_pct']
                    )
                    current_position.costs += entry_cost
                
                elif current_position is not None:
                    if curr_size == 0:
                        # Closing position
                        exit_cost = abs(current_position.size * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        current_position.costs += exit_cost
                        
                        # Calculate realized PnL
                        price_pnl = (curr_price - current_position.entry_price) * current_position.size
                        current_position.realized_pnl = price_pnl - current_position.costs
                        
                        # Record trade
                        self.record_trade({
                            'symbol': symbol,
                            'entry_time': current_position.entry_time,
                            'exit_time': timestamp,
                            'entry_price': current_position.entry_price,
                            'exit_price': curr_price,
                            'size': current_position.size,
                            'direction': current_position.direction,
                            'realized_pnl': current_position.realized_pnl,
                            'costs': current_position.costs
                        })
                        
                        current_position = None
                        self.current_positions.pop(symbol, None)
                    
                    elif not np.isclose(curr_size, current_position.size):
                        # Position size change
                        size_change = curr_size - current_position.size
                        
                        # Calculate costs for the adjustment
                        adjust_cost = abs(size_change * curr_price) * (
                            costs_config['transaction_fee_pct'] + 
                            costs_config['slippage_pct']
                        )
                        current_position.costs += adjust_cost
                        current_position.size = curr_size
                
                # Calculate unrealized PnL for current position
                if current_position is not None:
                    price_pnl = (curr_price - current_position.entry_price) * current_position.size
                    current_position.unrealized_pnl = price_pnl - current_position.costs
                    
                    # Add borrowing costs for short positions
                    if current_position.size < 0:
                        borrow_cost = (abs(current_position.size * curr_price) * 
                                     costs_config['borrowing_cost_pa'] / (24 * 252))
                        current_position.costs += borrow_cost
                
                # Record PnL snapshot
                pnl_entries.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'realized_pnl': current_position.realized_pnl if current_position else 0,
                    'unrealized_pnl': current_position.unrealized_pnl if current_position else 0,
                    'costs': current_position.costs if current_position else 0,
                    'net_pnl': ((current_position.realized_pnl + current_position.unrealized_pnl) 
                               if current_position else 0),
                    'capital_used': abs(curr_size * curr_price) if current_position else 0
                })
            
            # Update PnL history
            symbol_pnl = pd.DataFrame(pnl_entries)
            self.record_pnl({
                'timestamp': symbol_pnl['timestamp'],
                'symbol': symbol,
                'realized_pnl': symbol_pnl['realized_pnl'].sum(),
                'unrealized_pnl': symbol_pnl['unrealized_pnl'].sum(),
                'costs': symbol_pnl['costs'].sum(),
                'net_pnl': symbol_pnl['net_pnl'].sum(),
                'capital_used': symbol_pnl['capital_used'].sum()
            })
            
            # Calculate returns
            returns = symbol_pnl['net_pnl'].diff() / self.allocated_capital
            all_returns.append(returns)
        
        # Combine returns across symbols
        if all_returns:
            strategy_returns = pd.concat(all_returns, axis=1).sum(axis=1)
        else:
            strategy_returns = pd.Series(0, index=self.signals.index)
        
        return strategy_returns

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

    def update_allocation(self, new_capital: float, prices: Dict[str, float]):
        """
        Update position sizes based on new capital allocation
        
        Returns dictionary of required position adjustments:
        {symbol: (current_size, target_size)}
        """
        if not self.current_positions:
            self.allocated_capital = new_capital
            return {}
        
        position_adjustments = {}
        
        # Handle case where allocated_capital is zero
        if self.allocated_capital == 0:
            if new_capital > 0:
                # Opening new positions
                for symbol, current_size in self.current_positions.items():
                    if current_size != 0:
                        position_adjustments[symbol] = (0, current_size)
        else:
            # Normal case - adjust existing positions
            capital_ratio = new_capital / self.allocated_capital
            for symbol, current_size in self.current_positions.items():
                target_size = current_size * capital_ratio
                if not np.isclose(target_size, current_size):
                    position_adjustments[symbol] = (current_size, target_size)
        
        self.allocated_capital = new_capital
        return position_adjustments

class Portfolio:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies: Dict[str, Strategy] = {}
        self.allocations: Dict[str, float] = {}
        self.allocation_signs: Dict[str, int] = {}  # 1 for normal, -1 for inverse
        self.returns = pd.DataFrame()
        self.rebalance_history = []
        self.cash_allocation = 0.0
        self.cash_history = None
        
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
    
    def update_pnl_tracking(self, timestamp: pd.Timestamp):
        """Update PnL tracking at a given timestamp"""
        # Pre-allocate arrays for strategy metrics
        n_strategies = len(self.strategies)
        strategy_names = []
        allocated_capitals = np.zeros(n_strategies)
        realized_pnls = np.zeros(n_strategies)
        unrealized_pnls = np.zeros(n_strategies)
        total_costs = np.zeros(n_strategies)
        
        # Track strategy-level metrics with progress bar
        for i, (name, strategy) in enumerate(tqdm(self.strategies.items(), desc="Updating strategy metrics")):
            position_summary = strategy.get_position_summary()
            
            strategy_names.append(name)
            allocated_capitals[i] = strategy.allocated_capital
            realized_pnls[i] = position_summary['realized_pnl'].sum() if not position_summary.empty else 0
            unrealized_pnls[i] = position_summary['unrealized_pnl'].sum() if not position_summary.empty else 0
            total_costs[i] = position_summary['costs'].sum() if not position_summary.empty else 0
        
        # Update capital history using loc
        idx = len(self.capital_history)
        self.capital_history.loc[idx:idx+n_strategies-1, 'timestamp'] = timestamp
        self.capital_history.loc[idx:idx+n_strategies-1, 'strategy'] = strategy_names
        self.capital_history.loc[idx:idx+n_strategies-1, 'allocated_capital'] = allocated_capitals
        self.capital_history.loc[idx:idx+n_strategies-1, 'realized_pnl'] = realized_pnls
        self.capital_history.loc[idx:idx+n_strategies-1, 'unrealized_pnl'] = unrealized_pnls
        self.capital_history.loc[idx:idx+n_strategies-1, 'total_costs'] = total_costs
        
        # Calculate portfolio totals
        total_allocated = allocated_capitals.sum()
        total_realized = realized_pnls.sum()
        total_unrealized = unrealized_pnls.sum()
        total_costs_sum = total_costs.sum()
        cash_balance = self.current_capital * self.cash_allocation
        portfolio_value = total_allocated + total_realized + total_unrealized - total_costs_sum + cash_balance
        
        # Update total PnL using loc
        idx = len(self.total_pnl)
        self.total_pnl.loc[idx, 'timestamp'] = timestamp
        self.total_pnl.loc[idx, 'total_allocated'] = total_allocated
        self.total_pnl.loc[idx, 'total_realized_pnl'] = total_realized
        self.total_pnl.loc[idx, 'total_unrealized_pnl'] = total_unrealized
        self.total_pnl.loc[idx, 'total_costs'] = total_costs_sum
        self.total_pnl.loc[idx, 'cash_balance'] = cash_balance
        self.total_pnl.loc[idx, 'portfolio_value'] = portfolio_value
    
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
            'cash_allocation': self.cash_allocation,
            'position_adjustments': position_adjustments
        })
        
        # Update allocations
        self.allocations = new_allocations
        
        # Update PnL tracking after rebalance
        self.update_pnl_tracking(pd.Timestamp.now())
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict) -> pd.DataFrame:
        """Calculate returns and track PnL"""
        strategy_returns = {}
        
        # Calculate returns for each strategy
        for name, strategy in self.strategies.items():
            strategy_returns[name] = strategy.calculate_returns(
                price_data, 
                costs_config,
                allocation_sign=self.allocation_signs[name]
            )
        
        # Combine into DataFrame
        self.returns = pd.DataFrame(strategy_returns)
        
        # Initialize cash history
        self.cash_history = pd.Series(self.cash_allocation, index=self.returns.index)
        
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
                self.cash_history.loc[timestamp:] = rebalance['cash_allocation']
        
        # Calculate portfolio returns with time-varying weights
        for t in range(len(portfolio_returns)):
            weights = current_allocations.iloc[t]
            portfolio_returns.iloc[t] = (self.returns.iloc[t] * weights).sum()
        
        # Add cash returns (0) to returns DataFrame
        self.returns['Cash'] = 0.0
        self.returns['Portfolio'] = portfolio_returns
        
        # Update PnL tracking for each timestamp
        for timestamp in self.returns.index:
            self.update_pnl_tracking(timestamp)
        
        return self.returns
    
    def get_pnl_summary(self, start_date: Optional[pd.Timestamp] = None, 
                       end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Get PnL summary for a date range"""
        pnl = self.total_pnl.copy()
        
        if start_date:
            pnl = pnl[pnl['timestamp'] >= start_date]
        if end_date:
            pnl = pnl[pnl['timestamp'] <= end_date]
        
        # Calculate period metrics
        summary = pd.DataFrame({
            'Initial Value': [self.initial_capital],
            'Final Value': [pnl['portfolio_value'].iloc[-1]],
            'Total Return': [pnl['portfolio_value'].iloc[-1] / self.initial_capital - 1],
            'Realized PnL': [pnl['total_realized_pnl'].iloc[-1]],
            'Unrealized PnL': [pnl['total_unrealized_pnl'].iloc[-1]],
            'Total Costs': [pnl['total_costs'].iloc[-1]],
            'Current Cash': [pnl['cash_balance'].iloc[-1]]
        })
        
        return summary
    
    def get_strategy_pnl(self, strategy_name: str) -> pd.DataFrame:
        """Get detailed PnL history for a specific strategy"""
        return self.capital_history[
            self.capital_history['strategy'] == strategy_name
        ].set_index('timestamp')

class KellyPortfolio(Portfolio):
    def __init__(self, initial_capital: float = 1000000, lookback_days: int = 30):
        super().__init__(initial_capital)
        self.lookback_hours = lookback_days * 24
        self.bad_strategy_threshold = -0.5  # Kelly score below this is considered "bad"
    
    def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly Criterion for a return series (returns raw score)"""
        if len(returns) < self.lookback_hours:
            return 0.0
            
        recent_returns = returns.iloc[-self.lookback_hours:]
        
        wins = recent_returns > 0
        losses = recent_returns < 0
        
        if len(recent_returns[losses]) == 0:  # No losses
            return 1.0
        if len(recent_returns[wins]) == 0:  # No wins
            return -1.0
            
        win_prob = len(recent_returns[wins]) / len(recent_returns)
        avg_win = recent_returns[wins].mean() if len(recent_returns[wins]) > 0 else 0
        avg_loss = abs(recent_returns[losses].mean())
        
        if avg_loss == 0:  # Avoid division by zero
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        return kelly  # Return raw score (can be negative)
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Calculate optimal allocations and determine which strategies to inverse"""
        if not self.returns.empty:
            kelly_scores = {}
            
            # Calculate Kelly Criterion for each strategy
            for strategy_name in self.strategies:
                returns = self.returns[strategy_name]
                kelly = self.calculate_kelly_criterion(returns)
                kelly_scores[strategy_name] = kelly
                
                # Update allocation signs based on Kelly scores
                if kelly < self.bad_strategy_threshold:
                    # Strategy is bad enough to inverse
                    self.allocation_signs[strategy_name] = -1
                    kelly_scores[strategy_name] = abs(kelly)  # Use absolute value for allocation
                else:
                    self.allocation_signs[strategy_name] = 1
            
            # Filter for strategies with significant Kelly scores
            good_strategies = {k: v for k, v in kelly_scores.items() 
                             if abs(v) > 1}  # Use absolute value for filtering
            
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
                print("\nStrategy Allocations:")
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
        
        # Default to equal weight if no returns data yet
        return {k: 1.0/len(self.strategies) for k in self.strategies}
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict,
                         rebalance_frequency: str = '30D') -> pd.DataFrame:
        """Calculate returns and rebalance based on Kelly Criterion"""
        # First calculate initial returns with normal allocations
        super().calculate_returns(price_data, costs_config)
        
        # Get rebalancing dates
        dates = pd.date_range(
            start=self.returns.index[0],
            end=self.returns.index[-1],
            freq=rebalance_frequency
        )
        
        # Rebalance at each date
        for date in dates:
            if date in self.returns.index:
                # Get current prices for position adjustment
                current_prices = {
                    symbol: data.loc[date, 'Close'] 
                    for symbol, data in price_data.items()
                    if date in data.index
                }
                
                # Calculate and apply new allocations
                new_allocations = self.optimize_allocations()
                self.rebalance(new_allocations, current_prices)
                
                # Recalculate returns with proper signs
                for name, strategy in self.strategies.items():
                    if self.allocation_signs[name] != 0:
                        self.returns[name] = strategy.calculate_returns(
                            price_data, 
                            costs_config,
                            allocation_sign=self.allocation_signs[name]
                        )
        
        return self.returns

class BadStrategyPortfolio(Portfolio):
    def __init__(self, initial_capital: float = 1000000, lookback_days: int = 30):
        super().__init__(initial_capital)
        self.lookback_hours = lookback_days * 24
        self.bad_strategy_threshold = -0.5  # Threshold for identifying bad strategies
    
    def calculate_strategy_score(self, returns: pd.Series) -> float:
        """Calculate strategy score based on recent performance"""
        if len(returns) < self.lookback_hours:
            return 0.0
            
        recent_returns = returns.iloc[-self.lookback_hours:]
        
        # Calculate key metrics
        win_rate = (recent_returns > 0).mean()
        avg_win = recent_returns[recent_returns > 0].mean() if any(recent_returns > 0) else 0
        avg_loss = recent_returns[recent_returns < 0].mean() if any(recent_returns < 0) else 0
        sharpe = recent_returns.mean() / recent_returns.std() if recent_returns.std() != 0 else 0
        
        # Combine metrics into a score
        score = (win_rate - 0.5) + sharpe + (avg_win/abs(avg_loss) if avg_loss != 0 else 0)
        return score
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Find bad strategies to inverse"""
        if not self.returns.empty:
            strategy_scores = {}
            
            # Calculate scores for each strategy
            for strategy_name in self.strategies:
                returns = self.returns[strategy_name]
                score = self.calculate_strategy_score(returns)
                strategy_scores[strategy_name] = score
                
                # Update allocation signs - inverse bad strategies
                if score < self.bad_strategy_threshold:
                    self.allocation_signs[strategy_name] = -1
                    print(f"\nInversing {strategy_name} (Score: {score:.2f})")
                else:
                    self.allocation_signs[strategy_name] = 1
            
            # Equal weight allocation to all strategies
            n_strategies = len(self.strategies)
            allocations = {k: 1.0/n_strategies for k in self.strategies}
            self.cash_allocation = 0.0
            
            # Print allocation summary
            print("\nStrategy Allocations:")
            for strategy, alloc in allocations.items():
                direction = "INVERSE" if self.allocation_signs[strategy] < 0 else "NORMAL"
                print(f"{strategy}: {alloc:.1%} ({direction}) - Score: {strategy_scores[strategy]:.2f}")
            
            return allocations
        
        # Default to equal weight if no returns data yet
        return {k: 1.0/len(self.strategies) for k in self.strategies}
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict,
                         rebalance_frequency: str = '7D') -> pd.DataFrame:
        """Calculate returns and check for strategies to inverse weekly"""
        # First calculate initial returns
        super().calculate_returns(price_data, costs_config)
        
        # Get rebalancing dates
        dates = pd.date_range(
            start=self.returns.index[0],
            end=self.returns.index[-1],
            freq=rebalance_frequency
        )
        
        # Track strategy inversions
        self.inversion_history = pd.DataFrame(index=self.returns.index, 
                                            columns=self.strategies.keys(),
                                            data=1)
        
        # Rebalance and check for inversions at each date
        for date in dates:
            if date in self.returns.index:
                # Get current prices
                current_prices = {
                    symbol: data.loc[date, 'Close'] 
                    for symbol, data in price_data.items()
                    if date in data.index
                }
                
                # Update allocations and inversions
                new_allocations = self.optimize_allocations()
                self.rebalance(new_allocations, current_prices)
                
                # Record inversions
                for strategy in self.strategies:
                    self.inversion_history.loc[date:, strategy] = self.allocation_signs[strategy]
        
        # Recalculate returns with inversions
        strategy_returns = {}
        for name, strategy in self.strategies.items():
            # Apply time-varying inversions
            returns = pd.Series(index=self.returns.index)
            for t in range(len(returns)):
                returns.iloc[t] = strategy.calculate_returns(
                    price_data,
                    costs_config,
                    allocation_sign=self.inversion_history.loc[returns.index[t], name]
                ).iloc[t]
            strategy_returns[name] = returns
        
        self.returns = pd.DataFrame(strategy_returns)
        self.returns['Portfolio'] = sum(self.returns[s] * self.allocations[s] 
                                      for s in self.strategies)
        
        return self.returns

    def process_signals(self, signals: pd.DataFrame, price_data: Dict[str, pd.DataFrame]):
        """Process strategy signals and update positions"""
        # Group signals by timestamp to process in order
        grouped_signals = signals.groupby('timestamp')
        
        # Process signals in chronological order with progress bar
        for timestamp, timestamp_signals in tqdm(grouped_signals, desc="Processing strategy signals"):
            # Update positions based on signals
            for _, signal in timestamp_signals.iterrows():
                symbol = signal['symbol']
                signal_value = signal['signal']
                
                if symbol not in price_data:
                    continue
                    
                symbol_prices = price_data[symbol]
                if timestamp not in symbol_prices.index:
                    continue
                    
                current_price = symbol_prices.loc[timestamp, 'close']
                
                # Process the signal
                self._process_signal(timestamp, symbol, signal_value, current_price)
            
            # Update PnL tracking
            self.update_pnl_tracking(timestamp)