# %% [markdown]
#
# # Multi-Strategy Portfolio Construction

# %% Modules
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {SCRIPT_DIR}")

def load_signals(file_paths_config):
    """
    Load and format trading signals from multiple sources
    
    Parameters:
    -----------
    file_paths_config : list of dicts
        Each dict contains:
        - file_path: path to csv file (relative to script directory)
        - mapping_config: dict with column mappings and/or fixed values
    
    Returns:
    --------
    pd.DataFrame
        Combined signals dataframe with standardized format
    """
    signals = pd.DataFrame(columns=["timestamp", "strategy", "symbol", "signal"])
    
    for config in file_paths_config:
        abs_file_path = os.path.join(SCRIPT_DIR, config['file_path'])
        print(f"Loading signals from: {abs_file_path}")
        input_df = pd.read_csv(abs_file_path)
        temp_df = pd.DataFrame()
        
        for col in signals.columns:
            mapping = config['mapping_config'][col]
            if isinstance(mapping, dict) and 'from_column' in mapping:
                temp_df[col] = input_df[mapping['from_column']]
            elif isinstance(mapping, dict) and 'map_from' in mapping:
                temp_df[col] = input_df[mapping['map_from']].map(mapping['values'])
            else:
                temp_df[col] = mapping
        
        # Convert timestamp to datetime and localize to NY timezone
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        temp_df['timestamp'] = temp_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')
        
        signals = pd.concat([signals, temp_df], ignore_index=True)
    
    # Sort signals by timestamp
    signals = signals.sort_values('timestamp')
    print(f"Unique symbols in signals: {signals['symbol'].unique()}")
    return signals

def get_ticks(pair, grain='1H', directory='prices/'):
    """
    Load and process tick data for a given symbol
    """
    # Use script directory instead of workspace root for data directory
    abs_directory = os.path.join(SCRIPT_DIR, directory)  # Changed from WORKSPACE_ROOT to SCRIPT_DIR
    print(f"\nLooking for price data:")
    print(f"Symbol: {pair}")
    print(f"Directory: {abs_directory}")
    
    if not os.path.exists(abs_directory):
        raise FileNotFoundError(f"Data directory does not exist: {abs_directory}")
    
    # Build the glob pattern
    pattern = os.path.join(abs_directory, f'{pair}_Candlestick_1_Hour_*.csv')
    print(f"Searching with pattern: {pattern}")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"All files in directory:")
        all_files = os.listdir(abs_directory)
        for file in all_files:
            print(f"  - {file}")
        raise FileNotFoundError(f"No price data files found for {pair} in {abs_directory}")
    else:
        print(f"Found matching files: {matching_files}")
    
    # Load the CSV file
    file = matching_files[0]
    df = pd.read_csv(file)

    # Clean the 'Local time' column by removing the 'GMT-0500' part
    df['Local time'] = df['Local time'].str.replace(r' GMT[+-]\d{4}', '', regex=True)

    # Convert the cleaned 'Local time' column to datetime format
    df['Local time'] = pd.to_datetime(df['Local time'], format="%d.%m.%Y %H:%M:%S.%f")

    # Set 'Local time' as the index
    df.set_index('Local time', inplace=True)

    # Localize the index to 'America/New_York' timezone
    df.index = df.index.tz_localize('America/New_York', ambiguous='infer')

    # Resample to desired granularity (change 'H' to 'h')
    grain = grain.replace('H', 'h')  # Convert 'H' to 'h' for newer pandas versions
    df_r1 = df.resample(grain).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop any rows with NaN values and duplicates
    df_r1 = df_r1.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_r1 = df_r1[~df_r1.index.duplicated(keep='first')]
    df_r1['symbol'] = pair
    return df_r1

def get_price_data(symbols, grain='1H', directory='prices/'):
    """
    Fetch OHLC data and calculate returns for multiple symbols
    """
    price_data = {}
    failed_symbols = []
    print(f"\nFetching price data for symbols: {symbols}")
    
    for symbol in symbols:
        try:
            price_data[symbol] = get_ticks(symbol, grain=grain, directory=directory)
            price_data[symbol]['returns'] = price_data[symbol]['Close'].pct_change()
            print(f"Successfully loaded data for {symbol}")
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    if failed_symbols:
        print(f"\nWarning: Could not load data for the following symbols: {failed_symbols}")
    
    if not price_data:
        raise ValueError("No price data could be loaded for any symbols")
    
    return price_data

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
    pd.DataFrame
        DataFrame containing portfolio returns and weights
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

# %% Example usage:
signal_configs = [
    {
        'file_path': "trade_logs/all_trading_logs.csv",  # Path relative to script directory
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': {'from_column': 'Strategy'},
            'symbol': 'USA500.IDXUSD',
            'signal': {
                'map_from': 'Action',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    },
    {
        'file_path': "trade_logs/trade_log_jupyter.csv",  # Path relative to script directory
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': 'Momentum Long-Only',  # Fixed value
            'symbol': 'USDJPY',
            'signal': {
                'map_from': 'Type',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    }
]

# Load and process signals
signals = load_signals(signal_configs)

# Before calling get_price_data
unique_symbols = signals['symbol'].unique()
print("\nSymbols from signals:")
for symbol in unique_symbols:
    print(f"  - {symbol}")

# Get price data with correct directory
price_data = get_price_data(signals['symbol'].unique(), directory='prices/')

# Add this before calculating strategy returns
print("\nDebug information:")
for symbol in price_data:
    print(f"\n{symbol} price data:")
    print(f"Index timezone: {price_data[symbol].index.tz}")
    print(f"First timestamp: {price_data[symbol].index[0]}")
    print(f"Last timestamp: {price_data[symbol].index[-1]}")

print("\nSignals information:")
print(f"First timestamp: {signals['timestamp'].min()}")
print(f"Last timestamp: {signals['timestamp'].max()}")
print(f"Timestamp dtype: {signals['timestamp'].dtype}")

# Calculate strategy returns
strategy_returns = calculate_strategy_returns(signals, price_data)

# Add after strategy returns calculation:
print("\nCalculating portfolio returns and risk metrics...")

# Calculate portfolio returns
portfolio_values, weights_history, rebalance_dates = calculate_portfolio_returns(
    strategy_returns, 
    rebalance_threshold=0.05
)

# Calculate risk metrics for each strategy and the portfolio
risk_metrics = pd.DataFrame()

# Individual strategy metrics
for col in strategy_returns.columns:
    metrics = calculate_risk_metrics(strategy_returns[col], col)
    risk_metrics = pd.concat([risk_metrics, metrics], axis=1)

# Portfolio metrics
portfolio_metrics = calculate_risk_metrics(
    portfolio_values['returns'], 
    'Portfolio'
)
risk_metrics = pd.concat([risk_metrics, portfolio_metrics], axis=1)

# Print findings
print("\nRisk Metrics Summary:")
print(risk_metrics.round(4))

print(f"\nNumber of rebalances: {len(rebalance_dates)}")
print("Rebalancing dates:")
for date in rebalance_dates[:5]:
    print(f"  - {date}")
if len(rebalance_dates) > 5:
    print(f"  ... and {len(rebalance_dates)-5} more")

# Create visualizations
plt.style.use('default')  # Use default matplotlib style instead of seaborn
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Cumulative Returns
((1 + strategy_returns).cumprod()).plot(
    ax=axes[0,0], 
    title='Cumulative Strategy Returns'
)
((1 + portfolio_values['returns']).cumprod()).plot(
    ax=axes[0,0], 
    linewidth=3, 
    color='black', 
    label='Portfolio'
)
axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,0].set_ylabel('Cumulative Return')
axes[0,0].grid(True)

# 2. Strategy Correlations (replace seaborn heatmap with matplotlib)
corr_matrix = strategy_returns.corr()
im = axes[0,1].imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
axes[0,1].set_xticks(np.arange(len(corr_matrix.columns)))
axes[0,1].set_yticks(np.arange(len(corr_matrix.columns)))
axes[0,1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
axes[0,1].set_yticklabels(corr_matrix.columns)

# Add correlation values as text
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[0,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black")

axes[0,1].set_title('Strategy Correlations')
plt.colorbar(im, ax=axes[0,1])

# 3. Rolling Volatility (252-day)
strategy_returns.rolling(252).std().mul(np.sqrt(252)).plot(
    ax=axes[1,0], 
    title='Rolling Annualized Volatility'
)
axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].set_ylabel('Annualized Volatility')
axes[1,0].grid(True)

# 4. Strategy Weights Over Time
weights_history.plot(
    ax=axes[1,1], 
    title='Strategy Weights Over Time'
)
axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,1].set_ylabel('Weight')
axes[1,1].grid(True)

plt.tight_layout()
plt.show()
# %%
