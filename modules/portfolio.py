import pandas as pd
import numpy as np

def calculate_strategy_returns(signals, price_data):
    """
    Calculate returns for each strategy-symbol combination
    """
    strategy_returns = {}
    
    for strategy in signals['strategy'].unique():
        for symbol in signals[signals['strategy'] == strategy]['symbol'].unique():
            # Skip if we don't have price data for this symbol
            if symbol not in price_data or price_data[symbol] is None:
                print(f"Skipping {strategy}_{symbol} - no price data available")
                continue
                
            try:
                # Get signals for this strategy-symbol combination
                strat_signals = signals[
                    (signals['strategy'] == strategy) & 
                    (signals['symbol'] == symbol)
                ].copy()  # Create a copy to avoid SettingWithCopyWarning
                
                # Set timestamp as index
                strat_signals.set_index('timestamp', inplace=True)
                
                # Get corresponding price data
                symbol_prices = price_data[symbol]
                
                # Ensure both indexes are timezone-aware and aligned
                if strat_signals.index.tz != symbol_prices.index.tz:
                    strat_signals.index = strat_signals.index.tz_convert(symbol_prices.index.tz)
                
                # Align signals with returns using reindex and handle downcasting properly
                aligned_signals = (strat_signals['signal']
                    .reindex(symbol_prices.index)
                    .astype(float)  # Convert to float first
                    .ffill()  # Use ffill() instead of fillna(method='ffill')
                    .fillna(0))  # Fill remaining NaNs with 0
                
                # Calculate strategy returns
                strategy_returns[f"{strategy}_{symbol}"] = (
                    aligned_signals * symbol_prices['returns'].shift(-1)
                )
                print(f"Successfully calculated returns for {strategy}_{symbol}")
                
            except Exception as e:
                print(f"Error calculating returns for {strategy}_{symbol}: {str(e)}")
                continue
    
    if not strategy_returns:
        raise ValueError("No strategy returns could be calculated")
    
    return pd.DataFrame(strategy_returns)

def calculate_portfolio_returns(strategy_returns, rebalance_threshold=0.05):
    """
    Calculate portfolio returns with equal weight allocation and threshold rebalancing
    
    Parameters:
    -----------
    strategy_returns : pd.DataFrame
        DataFrame of strategy returns where columns are strategy_symbol combinations
    rebalance_threshold : float
        Threshold deviation from target weights that triggers rebalancing
    
    Returns:
    --------
    tuple
        (portfolio_values, weights_history, rebalance_dates)
    """
    # Initialize target weights (equal weight)
    n_strategies = len(strategy_returns.columns)
    target_weights = pd.Series(1/n_strategies, index=strategy_returns.columns)
    
    # Initialize portfolio tracking
    current_weights = target_weights.copy()
    portfolio_values = pd.DataFrame(index=strategy_returns.index)
    portfolio_values['values'] = 100  # Start with $100
    portfolio_values['returns'] = 0
    weights_history = pd.DataFrame(index=strategy_returns.index, columns=strategy_returns.columns)
    weights_history.iloc[0] = target_weights
    
    # Track rebalancing points
    rebalance_dates = [strategy_returns.index[0]]
    
    # Calculate cumulative portfolio value and track weights
    for i in range(1, len(strategy_returns)):
        # Get daily returns for each strategy
        daily_returns = strategy_returns.iloc[i]
        
        # Update strategy values and weights
        strategy_values = current_weights * (1 + daily_returns)
        total_value = strategy_values.sum()
        current_weights = strategy_values / total_value
        
        # Check if rebalancing is needed
        max_weight_diff = abs(current_weights - target_weights).max()
        if max_weight_diff > rebalance_threshold:
            current_weights = target_weights.copy()
            rebalance_dates.append(strategy_returns.index[i])
        
        # Store weights and calculate portfolio return
        weights_history.iloc[i] = current_weights
        portfolio_values.iloc[i, portfolio_values.columns.get_loc('returns')] = (
            (daily_returns * weights_history.iloc[i-1]).sum()
        )
        portfolio_values.iloc[i, portfolio_values.columns.get_loc('values')] = (
            portfolio_values.iloc[i-1]['values'] * (1 + portfolio_values.iloc[i]['returns'])
        )
    
    return portfolio_values, weights_history, rebalance_dates

def calculate_risk_metrics(returns_series, strategy_name=None):
    """
    Calculate key risk metrics for a return series
    """
    metrics = {}
    
    # Basic return metrics
    metrics['Total Return'] = (1 + returns_series).prod() - 1
    metrics['Ann. Return'] = (1 + returns_series).prod() ** (252/len(returns_series)) - 1
    metrics['Ann. Volatility'] = returns_series.std() * np.sqrt(252)
    
    # Risk metrics
    metrics['Sharpe Ratio'] = metrics['Ann. Return'] / metrics['Ann. Volatility']
    metrics['Max Drawdown'] = (1 + returns_series).cumprod().div(
        (1 + returns_series).cumprod().cummax()
    ).min() - 1
    
    # Additional metrics
    metrics['Win Rate'] = (returns_series > 0).mean()
    metrics['Profit Factor'] = abs(returns_series[returns_series > 0].sum() / 
                                 returns_series[returns_series < 0].sum())
    
    return pd.Series(metrics, name=strategy_name) 