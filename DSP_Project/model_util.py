import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class ForexDataProcessor:
    """
    Forex data processor for multi-timeframe analysis
    Innovation: Cross-currency training (EUR/USD) and testing (GBP/USD)
    """

    def __init__(self, pair, start_date, end_date):
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        self.data_1min = None
        self.data_5min = None

    def download_data(self):
        """Download forex data using yfinance"""
        print(f"Downloading {self.pair} data...")

        try:
            # Download 1-minute data
            ticker = yf.Ticker(f"{self.pair}=X")
            print(f"Ticker created: {self.pair}=X")

            self.data_1min = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="1m"
            )
            print(f"1-min data downloaded: {self.data_1min.shape}")

            # Download 5-minute data
            self.data_5min = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="5m"
            )
            print(f"5-min data downloaded: {self.data_5min.shape}")

            if self.data_1min.empty or self.data_5min.empty:
                print("WARNING: Downloaded data is empty!")
                return None, None

        except Exception as e:
            print(f"ERROR downloading data: {e}")
            return None, None

        return self.data_1min, self.data_5min

    def calculate_technical_indicators(self, df, window_12=12, window_24=24, window_36=36):
        """
        Calculate technical indicators
        Innovation: Added RSI and Bollinger Bands
        """
        df = df.copy()

        # Simple Moving Averages
        df['SMA_12'] = df['Close'].rolling(window=window_12).mean()
        df['SMA_24'] = df['Close'].rolling(window=window_24).mean()
        df['SMA_36'] = df['Close'].rolling(window=window_36).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=window_12, adjust=False).mean()
        df['EMA_24'] = df['Close'].ewm(span=window_24, adjust=False).mean()
        df['EMA_36'] = df['Close'].ewm(span=window_36, adjust=False).mean()

        # RSI (Innovation)
        rsi_indicator = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()

        # Bollinger Bands (Innovation)
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()

        return df

    def create_labels(self, df, k=5, threshold=0.0001):
        """
        Create labels using the paper's methodology
        l_t = (m+(t) - m-(t)) / m-(t)
        """
        df = df.copy()
        close_prices = df['Close'].values

        labels = []
        for t in range(len(close_prices)):
            if t < k or t >= len(close_prices) - k:
                labels.append(np.nan)
                continue

            # m-(t): mean of previous k prices
            m_minus = np.mean(close_prices[t - k:t])

            # m+(t): mean of next k prices
            m_plus = np.mean(close_prices[t + 1:t + k + 1])

            # Calculate l_t
            l_t = (m_plus - m_minus) / m_minus

            # Binary classification: 1 (up), 0 (down)
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
        """
        Create training windows
        - 32 consecutive 5-minute candlesticks (160 minutes)
        - 32 consecutive 1-minute candlesticks (last 32 minutes)
        """
        features_5min = ['Open', 'High', 'Low', 'Close',
                         'EMA_12', 'SMA_12', 'EMA_24', 'EMA_36',
                         'RSI', 'BB_mid']

        features_1min = ['Open', 'High', 'Low', 'Close',
                         'EMA_12', 'SMA_12', 'EMA_24', 'EMA_36',
                         'RSI', 'BB_mid']

        # Drop NaN values
        df_5min = df_5min.dropna()
        df_1min = df_1min.dropna()

        X_5min_list = []
        X_1min_list = []
        y_list = []

        for i in range(window_size, len(df_5min)):
            # Get 5-minute window
            window_5min = df_5min.iloc[i - window_size:i][features_5min].values

            # Get corresponding 1-minute window (last 32 minutes)
            time_5min = df_5min.index[i]
            mask_1min = (df_1min.index >= time_5min - pd.Timedelta(minutes=32)) & \
                        (df_1min.index < time_5min)

            window_1min = df_1min[mask_1min][features_1min].values

            if len(window_1min) != window_size:
                continue

            # Normalize windows
            window_5min_norm = self.normalize_data(window_5min.T)
            window_1min_norm = self.normalize_data(window_1min.T)

            X_5min_list.append(window_5min_norm)
            X_1min_list.append(window_1min_norm)
            y_list.append(df_5min.iloc[i]['Label'])

        return np.array(X_5min_list), np.array(X_1min_list), np.array(y_list)

    def process_pipeline(self):
        """Complete preprocessing pipeline"""
        # Download data
        df_1min, df_5min = self.download_data()

        # Calculate indicators
        df_1min = self.calculate_technical_indicators(df_1min)
        df_5min = self.calculate_technical_indicators(df_5min)

        # Create labels
        df_5min = self.create_labels(df_5min)

        # Create windows
        X_5min, X_1min, y = self.create_windows(df_5min, df_1min)

        print(f"\nFinal dataset shapes:")
        print(f"X_5min: {X_5min.shape}")
        print(f"X_1min: {X_1min.shape}")
        print(f"y: {y.shape}")
        print(f"Label distribution: {np.bincount(y.astype(int))}")

        return X_5min, X_1min, y


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("SCRIPT STARTED - Data Processing")
    print("=" * 50)

    # Training data: EUR/USD
    print("\n1. Processing EUR/USD training data...")
    # Training data: EUR/USD
    processor_train = ForexDataProcessor(
        pair="EURUSD",
        start_date="2025-11-05",
        end_date="2025-11-06"
    )
    X_5min_train, X_1min_train, y_train = processor_train.process_pipeline()

    # Save training data
    np.save('X_5min_train_eurusd.npy', X_5min_train)
    np.save('X_1min_train_eurusd.npy', X_1min_train)
    np.save('y_train_eurusd.npy', y_train)

    # Testing data: GBP/USD (Innovation: Cross-currency testing)
    processor_test = ForexDataProcessor(
        pair="GBPUSD",
        start_date="2025-11-05",
        end_date="2025-11-06"
    )
    X_5min_test, X_1min_test, y_test = processor_test.process_pipeline()

    # Save testing data
    np.save('X_5min_test_gbpusd.npy', X_5min_test)
    np.save('X_1min_test_gbpusd.npy', X_1min_test)
    np.save('y_test_gbpusd.npy', y_test)

    print("\n✓ Data preprocessing complete!")