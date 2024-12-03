# %% [markdown]
#
# # Multi-Strategy Portfolio Construction

# %% Modules
import pandas as pd
import glob  # Add missing import for glob module
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent directory of the script directory
print(f"Script directory: {SCRIPT_DIR}")
print(f"Workspace root: {WORKSPACE_ROOT}")

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
        
        signals = pd.concat([signals, temp_df], ignore_index=True)
    
    print(f"Unique symbols in signals: {signals['symbol'].unique()}")
    return signals

def get_ticks(pair, grain='1H', directory='prices/'):
    """
    Load and process tick data for a given symbol
    """
    # Use workspace root for data directory
    abs_directory = os.path.join(WORKSPACE_ROOT, directory)
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

    # Resample to desired granularity
    df_r1 = df.resample(grain).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop any rows with NaN values
    df_r1 = df_r1.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_r1['asset'] = pair
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
                
            # Get signals for this strategy-symbol combination
            strat_signals = signals[
                (signals['strategy'] == strategy) & 
                (signals['symbol'] == symbol)
            ].set_index('timestamp')
            
            # Get corresponding price data
            symbol_prices = price_data[symbol]
            
            try:
                # Align signals with returns
                aligned_signals = strat_signals['signal'].reindex(
                    symbol_prices.index, 
                    method='ffill'
                ).fillna(0)
                
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

# Calculate strategy returns
strategy_returns = calculate_strategy_returns(signals, price_data)
# %%
