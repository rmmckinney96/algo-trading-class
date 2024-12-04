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
        
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], costs_config: Dict, allocation_sign: int = 1) -> pd.Series:
        """
        Calculate strategy returns using price data and signals
        
        Parameters:
        -----------
        allocation_sign : int
            1 for normal allocation, -1 for inverse positions
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
            # Apply allocation sign to signals
            positions['signal'] = symbol_signals.reindex(prices.index).ffill().fillna(0) * allocation_sign
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
    
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame], 
                         costs_config: Dict) -> pd.DataFrame:
        """Calculate returns for all strategies and portfolio"""
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
        
        return self.returns

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