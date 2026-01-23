import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class BitcoinDataProcessor:
    """Bitcoin data processor for multi-timeframe analysis"""

    def __init__(self, csv_path_1min=None, csv_path_5min=None, start_date=None, end_date=None):
        self.csv_path_1min = csv_path_1min
        self.csv_path_5min = csv_path_5min
        self.start_date = start_date
        self.end_date = end_date
        self.data_1min = None
        self.data_5min = None

    def load_csv_data(self, csv_path):
        """Load Bitcoin data from CSV file"""
        print(f"Loading data from {csv_path}...")

        try:
            # Try reading with different separators
            df = None

            # First, try semicolon separator (for your CoinMarketCap format)
            try:
                df = pd.read_csv(csv_path, sep=';')
                print(f"✓ Loaded with semicolon separator")
            except:
                pass

            # If that didn't work, try comma separator
            if df is None or len(df.columns) == 1:
                try:
                    df = pd.read_csv(csv_path, sep=',')
                    print(f"✓ Loaded with comma separator")
                except:
                    pass

            if df is None:
                raise ValueError("Could not read CSV with any separator")

            print(f"Raw CSV shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            # Show first few rows for debugging
            print(f"\nFirst row sample:")
            print(df.head(1).to_dict('records'))

            # Column name variations
            timestamp_cols = ['timestamp', 'Timestamp', 'timeOpen', 'Date', 'date', 'time', 'Time', 'datetime',
                              'Datetime']
            open_cols = ['open', 'Open', 'Open_Price']
            high_cols = ['high', 'High', 'High_Price']
            low_cols = ['low', 'Low', 'Low_Price']
            close_cols = ['close', 'Close', 'Close_Price', 'Price']
            volume_cols = ['volume', 'Volume', 'Volume_(BTC)', 'Volume_(Currency)']

            timestamp_col = next((col for col in timestamp_cols if col in df.columns), None)
            open_col = next((col for col in open_cols if col in df.columns), None)
            high_col = next((col for col in high_cols if col in df.columns), None)
            low_col = next((col for col in low_cols if col in df.columns), None)
            close_col = next((col for col in close_cols if col in df.columns), None)
            volume_col = next((col for col in volume_cols if col in df.columns), None)

            if not timestamp_col:
                raise ValueError(f"Could not find timestamp column. Available: {df.columns.tolist()}")

            print(f"\n📋 Detected columns:")
            print(f"   Timestamp: {timestamp_col}")
            print(f"   Open: {open_col}")
            print(f"   High: {high_col}")
            print(f"   Low: {low_col}")
            print(f"   Close: {close_col}")
            print(f"   Volume: {volume_col}")

            df_std = pd.DataFrame()

            # Convert timestamp - handle both Unix timestamps and datetime strings
            try:
                # Try Unix timestamp first
                df_std['Datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
                print(f"   ✓ Parsed as Unix timestamp (milliseconds)")
            except:
                try:
                    df_std['Datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
                    print(f"   ✓ Parsed as Unix timestamp (seconds)")
                except:
                    try:
                        df_std['Datetime'] = pd.to_datetime(df[timestamp_col])
                        print(f"   ✓ Parsed as datetime string")
                    except Exception as e:
                        raise ValueError(f"Could not parse timestamp column: {e}")

            # Convert price columns
            if open_col:
                df_std['Open'] = pd.to_numeric(df[open_col], errors='coerce')
            if high_col:
                df_std['High'] = pd.to_numeric(df[high_col], errors='coerce')
            if low_col:
                df_std['Low'] = pd.to_numeric(df[low_col], errors='coerce')
            if close_col:
                df_std['Close'] = pd.to_numeric(df[close_col], errors='coerce')
            else:
                raise ValueError("Could not find Close price column")

            # Fill missing OHLC with Close
            if 'Open' not in df_std.columns:
                df_std['Open'] = df_std['Close']
            if 'High' not in df_std.columns:
                df_std['High'] = df_std['Close']
            if 'Low' not in df_std.columns:
                df_std['Low'] = df_std['Close']

            if volume_col:
                df_std['Volume'] = pd.to_numeric(df[volume_col], errors='coerce')
            else:
                df_std['Volume'] = 0

            # Remove rows with NaN in critical columns
            df_std = df_std.dropna(subset=['Close'])

            df_std.set_index('Datetime', inplace=True)
            df_std.sort_index(inplace=True)

            # Filter by date range if provided
            if self.start_date:
                df_std = df_std[df_std.index >= self.start_date]
            if self.end_date:
                df_std = df_std[df_std.index <= self.end_date]

            print(f"\n✓ Processed data shape: {df_std.shape}")
            print(f"✓ Date range: {df_std.index.min()} to {df_std.index.max()}")

            # Calculate time interval
            if len(df_std) > 1:
                time_diff = (df_std.index[1] - df_std.index[0]).total_seconds() / 60
                print(f"✓ Detected interval: ~{time_diff:.1f} minutes")

            return df_std

        except Exception as e:
            print(f"❌ ERROR loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None

    def resample_to_5min(self, df_1min):
        """Resample to 5-minute intervals"""
        print("\n🔄 Resampling to 5-minute intervals...")

        # Determine current interval
        if len(df_1min) > 1:
            interval_mins = (df_1min.index[1] - df_1min.index[0]).total_seconds() / 60
            print(f"   Current interval: ~{interval_mins:.1f} minutes")

            if interval_mins >= 5:
                print(f"   ⚠️ Data is already at {interval_mins:.1f}-minute intervals")
                print(f"   Using as-is for 5-minute timeframe")
                return df_1min.copy()

        df_5min = df_1min.resample('5min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        print(f"   ✓ 5-minute data shape: {df_5min.shape}")
        return df_5min

    def resample_to_1min(self, df):
        """Resample to 1-minute intervals if needed"""
        if len(df) < 2:
            return df

        interval_mins = (df.index[1] - df.index[0]).total_seconds() / 60

        if interval_mins <= 1:
            return df

        print(f"\n🔄 Resampling from {interval_mins:.1f}-min to 1-min intervals...")

        df_1min = df.resample('1min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).interpolate(method='linear')

        print(f"   ✓ 1-minute data shape: {df_1min.shape}")
        return df_1min

    def download_data(self):
        """Load Bitcoin data from CSV files"""
        print("\n📂 Loading Bitcoin data from CSV files...")

        if not self.csv_path_1min or not os.path.exists(self.csv_path_1min):
            print(f"❌ ERROR: CSV file not found: {self.csv_path_1min}")
            return None, None

        data = self.load_csv_data(self.csv_path_1min)

        if data is None or data.empty:
            print("❌ ERROR: Failed to load data")
            return None, None

        # Check if we need to resample to 1-minute
        if len(data) > 1:
            interval_mins = (data.index[1] - data.index[0]).total_seconds() / 60
            if interval_mins > 1:
                self.data_1min = self.resample_to_1min(data)
            else:
                self.data_1min = data
        else:
            self.data_1min = data

        # Create 5-minute data
        if self.csv_path_5min and os.path.exists(self.csv_path_5min):
            self.data_5min = self.load_csv_data(self.csv_path_5min)
        else:
            self.data_5min = self.resample_to_5min(self.data_1min)

        if self.data_5min is None or self.data_5min.empty:
            print("❌ ERROR: Failed to create 5-minute data")
            return None, None

        return self.data_1min, self.data_5min

    def calculate_technical_indicators(self, df, window_12=12, window_24=24, window_36=36):
        """Calculate technical indicators"""
        df = df.copy()

        df['SMA_12'] = df['Close'].rolling(window=window_12).mean()
        df['SMA_24'] = df['Close'].rolling(window=window_24).mean()
        df['SMA_36'] = df['Close'].rolling(window=window_36).mean()

        df['EMA_12'] = df['Close'].ewm(span=window_12, adjust=False).mean()
        df['EMA_24'] = df['Close'].ewm(span=window_24, adjust=False).mean()
        df['EMA_36'] = df['Close'].ewm(span=window_36, adjust=False).mean()

        rsi_indicator = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()

        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()

        return df

    def create_labels(self, df, k=5, threshold=0.0001):
        """Create labels using paper's methodology"""
        df = df.copy()
        close_prices = df['Close'].values

        labels = []
        for t in range(len(close_prices)):
            if t < k or t >= len(close_prices) - k:
                labels.append(np.nan)
                continue

            m_minus = np.mean(close_prices[t - k:t])
            m_plus = np.mean(close_prices[t + 1:t + k + 1])
            l_t = (m_plus - m_minus) / m_minus
            label = 1 if l_t > threshold else 0
            labels.append(label)

        df['Label'] = labels
        return df

    def normalize_data(self, data):
        """Normalize each row independently to [-1, 1]"""
        normalized = np.zeros_like(data)
        for i in range(data.shape[0]):
            row = data[i]
            min_val = np.min(row)
            max_val = np.max(row)
            if max_val - min_val > 0:
                normalized[i] = 2 * (row - min_val) / (max_val - min_val) - 1
            else:
                normalized[i] = np.zeros_like(row)
        return normalized

    def create_windows(self, df_5min, df_1min, window_size=32):
        """Create training windows"""
        features_5min = ['Open', 'High', 'Low', 'Close',
                         'EMA_12', 'SMA_12', 'EMA_24', 'EMA_36',
                         'RSI', 'BB_mid']

        features_1min = ['Open', 'High', 'Low', 'Close',
                         'EMA_12', 'SMA_12', 'EMA_24', 'EMA_36',
                         'RSI', 'BB_mid']

        df_5min = df_5min.dropna()
        df_1min = df_1min.dropna()

        X_5min_list_raw = []
        X_1min_list_raw = []
        X_5min_list = []
        X_1min_list = []
        y_list = []

        for i in range(window_size, len(df_5min)):
            window_5min = df_5min.iloc[i - window_size:i][features_5min].values
            time_5min = df_5min.index[i]
            mask_1min = (df_1min.index >= time_5min - pd.Timedelta(minutes=32)) & \
                        (df_1min.index < time_5min)
            window_1min = df_1min[mask_1min][features_1min].values

            if len(window_1min) != window_size:
                continue

            X_5min_list_raw.append(window_5min)
            X_1min_list_raw.append(window_1min)

            window_5min_norm = self.normalize_data(window_5min.T)
            window_1min_norm = self.normalize_data(window_1min.T)

            X_5min_list.append(window_5min_norm)
            X_1min_list.append(window_1min_norm)
            y_list.append(df_5min.iloc[i]['Label'])

        return (np.array(X_5min_list_raw), np.array(X_1min_list_raw),
                np.array(X_5min_list), np.array(X_1min_list),
                np.array(y_list))

    def process_pipeline(self):
        """Complete preprocessing pipeline"""
        df_1min, df_5min = self.download_data()

        if df_1min is None or df_5min is None:
            return None, None, None, None, None

        print("\n📊 Calculating technical indicators...")
        df_1min = self.calculate_technical_indicators(df_1min)
        df_5min = self.calculate_technical_indicators(df_5min)

        print("🏷️  Creating labels...")
        df_5min = self.create_labels(df_5min)

        print("🪟 Creating sliding windows...")
        X_5min_raw, X_1min_raw, X_5min, X_1min, y = self.create_windows(df_5min, df_1min)

        print(f"\n📈 Final dataset shapes:")
        print(f"   X_5min (normalized): {X_5min.shape}")
        print(f"   X_1min (normalized): {X_1min.shape}")
        print(f"   X_5min_raw (unscaled): {X_5min_raw.shape}")
        print(f"   X_1min_raw (unscaled): {X_1min_raw.shape}")
        print(f"   y (labels): {y.shape}")
        print(f"   Label distribution: {np.bincount(y.astype(int))}")

        return X_5min_raw, X_1min_raw, X_5min, X_1min, y


def find_csv_files(directory="."):
    """Find all CSV files in directory and filter for Bitcoin data"""
    all_csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # Filter out known non-Bitcoin files
    bitcoin_files = []
    for f in all_csv_files:
        basename = os.path.basename(f).lower()
        # Skip files that are clearly not Bitcoin data
        if any(skip in basename for skip in ['result', 'metric', 'performance', 'accuracy', 'model']):
            continue
        bitcoin_files.append(f)

    return bitcoin_files


def inspect_csv_file(csv_path, num_rows=5):
    """Inspect a CSV file to check if it's Bitcoin data"""
    try:
        # Try different separators
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(csv_path, sep=sep, nrows=num_rows)
                if len(df.columns) > 1:
                    break
            except:
                continue

        # Check if it has Bitcoin-like columns
        cols_lower = [col.lower() for col in df.columns]
        has_price = any(word in ' '.join(cols_lower) for word in ['close', 'price', 'open', 'high', 'low'])
        has_time = any(word in ' '.join(cols_lower) for word in ['time', 'date', 'timestamp'])

        return has_price and has_time, df.columns.tolist(), len(df.columns)
    except:
        return False, [], 0


def main():
    print("=" * 70)
    print("  BITCOIN DATA PROCESSOR - Semicolon CSV Support")
    print("=" * 70)

    # Search for CSV files
    print("\n🔍 Searching for Bitcoin CSV files in current directory...")
    print(f"📍 Directory: {os.getcwd()}\n")

    csv_files = find_csv_files()

    if not csv_files:
        print("❌ No CSV files found in current directory!")
        print("\n📋 INSTRUCTIONS:")
        print("1. Place your Bitcoin CSV file in this directory")
        print("2. Make sure filename contains 'bitcoin' or 'btc'")
        print("3. Run this script again")
        return

    # Inspect each file and filter for Bitcoin data
    print("🔎 Inspecting CSV files...")
    valid_files = []
    for f in csv_files:
        is_bitcoin, columns, num_cols = inspect_csv_file(f)
        filename = os.path.basename(f)
        size_mb = os.path.getsize(f) / (1024 * 1024)

        if is_bitcoin:
            print(f"   ✅ {filename} ({size_mb:.2f} MB, {num_cols} columns)")
            print(f"      Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
            valid_files.append(f)
        else:
            print(f"   ⏭️  {filename} - Skipped (not Bitcoin data)")

    if not valid_files:
        print("\n❌ No valid Bitcoin CSV files found!")
        print("\n💡 Looking for files with columns like:")
        print("   - timestamp, date, time")
        print("   - open, high, low, close, price")
        print("\n📋 Files found but skipped:")
        for f in csv_files:
            _, cols, _ = inspect_csv_file(f)
            print(f"   • {os.path.basename(f)}")
            print(f"     Columns: {', '.join(cols)}")
        return

    print(f"\n✅ Found {len(valid_files)} valid Bitcoin CSV file(s)")

    # Auto-select if only one file
    if len(valid_files) == 1:
        selected_file = valid_files[0]
        print(f"\n🎯 Auto-selecting: {os.path.basename(selected_file)}")
    else:
        print("\n📝 Select a file:")
        for i, f in enumerate(valid_files, 1):
            print(f"   {i}. {os.path.basename(f)}")
        choice = input("\nEnter number (or press Enter for file #1): ").strip()
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            selected_file = valid_files[idx]
        except (ValueError, IndexError):
            print("⚠️  Invalid choice, using first file")
            selected_file = valid_files[0]

    print(f"\n🚀 Processing: {os.path.basename(selected_file)}")
    print("-" * 70)

    # Process the data
    processor = BitcoinDataProcessor(
        csv_path_1min=selected_file,
        csv_path_5min=None,
        start_date=None,
        end_date=None
    )

    result = processor.process_pipeline()

    if result[0] is not None:
        X_5min_raw, X_1min_raw, X_5min, X_1min, y = result

        # Save processed data
        np.save('X_5min_btc_raw.npy', X_5min_raw)
        np.save('X_1min_btc_raw.npy', X_1min_raw)
        np.save('X_5min_btc.npy', X_5min)
        np.save('X_1min_btc.npy', X_1min)
        np.save('y_btc.npy', y)

        print("\n" + "=" * 70)
        print("  ✅ SUCCESS! Bitcoin data preprocessing complete!")
        print("=" * 70)
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(y):,}")
        print(f"   Up movements (1): {np.sum(y == 1):,} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
        print(f"   Down movements (0): {np.sum(y == 0):,} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
        print(f"\n💾 Files saved in: {os.getcwd()}")
        print("   ✓ X_5min_btc_raw.npy (raw 5-min data)")
        print("   ✓ X_1min_btc_raw.npy (raw 1-min data)")
        print("   ✓ X_5min_btc.npy (normalized 5-min)")
        print("   ✓ X_1min_btc.npy (normalized 1-min)")
        print("   ✓ y_btc.npy (labels)")
        print("\n✨ Ready for model training!")
    else:
        print("\n" + "=" * 70)
        print("  ❌ Failed to process Bitcoin data")
        print("=" * 70)
        print("\n🔧 Troubleshooting:")
        print("1. Check the error messages above")
        print("2. Verify the CSV has timestamp and price columns")
        print("3. Ensure the CSV is properly formatted")


if __name__ == "__main__":
    main()
