# %% [markdown]
#
# # Multi-Strategy Portfolio Construction

# %% Modules
import pandas as pd
import glob  # Add missing import for glob module

def load_signals(file_paths_config):
    """
    Load and format trading signals from multiple sources
    
    Parameters:
    -----------
    file_paths_config : list of dicts
        Each dict contains:
        - file_path: path to csv file
        - mapping_config: dict with column mappings and/or fixed values
    
    Returns:
    --------
    pd.DataFrame
        Combined signals dataframe with standardized format
    """
    signals = pd.DataFrame(columns=["timestamp", "strategy", "symbol", "signal"])
    
    for config in file_paths_config:
        input_df = pd.read_csv(config['file_path'])
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
    
    return signals


def get_ticks(pair, grain='1H', directory='data/'):
    # Step 1: Load the CSV file without parsing the dates initially
    file = glob.glob(f'{directory}{pair}_Candlestick_1_Hour_*.csv')[0]
    df = pd.read_csv(file,)

    # Step 2: Clean the 'Local time' column by removing the 'GMT-0500' part
    df['Local time'] = df['Local time'].str.replace(r' GMT[+-]\d{4}', '', regex=True)

    # Step 3: Convert the cleaned 'Local time' column to datetime format
    df['Local time'] = pd.to_datetime(df['Local time'], format="%d.%m.%Y %H:%M:%S.%f")

    # Step 4: Set 'Local time' as the index
    df.set_index('Local time', inplace=True)

    # Step 5: Localize the index to 'America/New_York' timezone, handling ambiguous DST times
    df.index = df.index.tz_localize('America/New_York', ambiguous='infer')

    # Resample to 4-hour bars
    df_r1 = df.resample(grain).agg({
        'Open': 'first',   # Take the first value in the window
        'High': 'max',     # Take the maximum value in the window
        'Low': 'min',      # Take the minimum value in the window
        'Close': 'last',   # Take the last value in the window
        'Volume': 'sum'    # Sum the volume over the window
    })

    # Drop any rows where Open, High, Low, or Close is NaN (which could happen due to gaps)
    df_r1 = df_r1.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_r1['asset'] = pair
    return df_r1

def get_price_data(symbols, grain='1H', directory='data/'):
    """
    Fetch OHLC data and calculate returns for multiple symbols
    
    Parameters:
    -----------
    symbols : list
        List of symbol strings
    grain : str
        Time granularity for the data
    directory : str
        Directory path for data storage
    
    Returns:
    --------
    dict
        Dictionary of dataframes containing OHLC and returns data for each symbol
    """
    price_data = {}
    
    for symbol in symbols:
        price_data[symbol] = get_ticks(symbol, grain=grain, directory=directory)
        price_data[symbol]['returns'] = price_data[symbol]['close'].pct_change()
    
    return price_data

def calculate_strategy_returns(signals, price_data):
    """
    Calculate returns for each strategy-symbol combination
    
    Parameters:
    -----------
    signals : pd.DataFrame
        DataFrame containing trading signals
    price_data : dict
        Dictionary of price data for each symbol
    
    Returns:
    --------
    pd.DataFrame
        Combined returns for all strategy-symbol combinations
    """
    strategy_returns = {}
    
    for strategy in signals['strategy'].unique():
        for symbol in signals[signals['strategy'] == strategy]['symbol'].unique():
            # Get signals for this strategy-symbol combination
            strat_signals = signals[
                (signals['strategy'] == strategy) & 
                (signals['symbol'] == symbol)
            ].set_index('timestamp')
            
            # Get corresponding price data
            symbol_prices = price_data[symbol]
            
            # Align signals with returns
            aligned_signals = strat_signals['signal'].reindex(
                symbol_prices.index, 
                method='ffill'
            ).fillna(0)
            
            # Calculate strategy returns
            strategy_returns[f"{strategy}_{symbol}"] = (
                aligned_signals * symbol_prices['returns'].shift(-1)
            )
    
    return pd.DataFrame(strategy_returns)

# %% Example usage:
signal_configs = [
    {
        'file_path': "project/trade_logs/all_trading_logs.csv",
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': {'from_column': 'Strategy'},
            'symbol': 'SPY',  # Fixed value since it's not in the CSV
            'signal': {
                'map_from': 'Action',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    },
    {
        'file_path': "project/trade_logs/trade_log_jupyter.csv",
        'mapping_config': {
            'timestamp': {'from_column': 'Date'},
            'strategy': 'Momentum Long-Only',  # Fixed value
            'symbol': 'USDJPY',  # Fixed value
            'signal': {
                'map_from': 'Type',
                'values': {'Buy': 1, 'Sell': 0}
            }
        }
    }
]

# Load and process signals
signals = load_signals(signal_configs)

# Get price data
price_data = get_price_data(signals['symbol'].unique())

# Calculate strategy returns
strategy_returns = calculate_strategy_returns(signals, price_data)
# %%
