import pandas as pd
import glob
import os

def load_signals(file_paths_config, script_dir):
    """Load and format trading signals from multiple sources"""
    signals = pd.DataFrame(columns=["timestamp", "strategy", "symbol", "signal"])
    
    for config in file_paths_config:
        abs_file_path = os.path.join(script_dir, config['file_path'])
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

def get_ticks(pair, script_dir, grain='1H', directory='prices/'):
    """Load and process tick data for a given symbol"""
    # Implementation remains the same but uses script_dir parameter
    ...

def get_price_data(symbols, script_dir, grain='1H', directory='prices/'):
    """Fetch OHLC data and calculate returns for multiple symbols"""
    # Implementation remains the same but uses script_dir parameter
    ... 