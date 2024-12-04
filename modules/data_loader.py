import pandas as pd
import glob
import os
from typing import Dict, List

def load_signals(file_paths_config: List[Dict]) -> pd.DataFrame:
    """
    Load and format trading signals from multiple sources
    """
    # Initialize signals DataFrame with proper dtypes
    signals = pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns]'),
        'strategy': pd.Series(dtype='str'),
        'symbol': pd.Series(dtype='str'),
        'signal': pd.Series(dtype='float64')
    })
    
    for config in file_paths_config:
        abs_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['file_path'])
        print(f"Loading signals from: {abs_file_path}")
        input_df = pd.read_csv(abs_file_path)
        
        # Create temp_df with proper dtypes
        temp_df = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'strategy': pd.Series(dtype='str'),
            'symbol': pd.Series(dtype='str'),
            'signal': pd.Series(dtype='float64')
        })
        
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
        
        # Ensure signal is float
        temp_df['signal'] = temp_df['signal'].astype(float)
        
        signals = pd.concat([signals, temp_df], ignore_index=True)
    
    # Sort signals by timestamp
    signals = signals.sort_values('timestamp')
    return signals

def get_ticks(pair: str, grain: str = '1H', directory: str = 'prices/') -> pd.DataFrame:
    """
    Load and process tick data for a given symbol
    """
    abs_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)
    print(f"\nLooking for price data:")
    print(f"Symbol: {pair}")
    print(f"Directory: {abs_directory}")
    
    if not os.path.exists(abs_directory):
        raise FileNotFoundError(f"Data directory does not exist: {abs_directory}")
    
    # Build the glob pattern
    pattern = os.path.join(abs_directory, f'{pair}_Candlestick_1_Hour_*.csv')
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No price data files found for {pair} in {abs_directory}")
    
    # Load the CSV file
    file = matching_files[0]
    df = pd.read_csv(file)

    # Clean the 'Local time' column and convert to timestamp
    df['Local time'] = df['Local time'].str.replace(r' GMT[+-]\d{4}', '', regex=True)
    df['timestamp'] = pd.to_datetime(df['Local time'], format="%d.%m.%Y %H:%M:%S.%f")
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize('America/New_York', ambiguous='infer')

    # Resample to desired granularity
    grain = grain.replace('H', 'h')  # Convert 'H' to 'h' for newer pandas versions
    df_r1 = df.resample(grain).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Clean up the data
    df_r1 = df_r1.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_r1 = df_r1[~df_r1.index.duplicated(keep='first')]
    df_r1['symbol'] = pair
    return df_r1

def get_price_data(symbols: List[str], grain: str = '1H', directory: str = 'prices/') -> Dict[str, pd.DataFrame]:
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