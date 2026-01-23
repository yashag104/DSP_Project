import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FFTConv1D(layers.Layer):
    """
    1D Convolution using FFT (Frequency Domain)
    Innovation: Compare FFT vs standard convolution performance
    """

    def __init__(self, filters, kernel_size, **kwargs):
        super(FFTConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Get input dimensions
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Pad for FFT
        fft_size = seq_len + self.kernel_size - 1
        fft_size = 2 ** int(np.ceil(np.log2(fft_size)))

        # Pad input
        inputs_padded = tf.pad(inputs, [[0, 0], [0, fft_size - seq_len], [0, 0]])

        # Apply FFT to input
        inputs_fft = tf.signal.rfft(inputs_padded, fft_length=[fft_size])

        outputs = []
        for i in range(self.filters):
            filter_outputs = []
            for j in range(inputs.shape[-1]):
                # Pad kernel
                kernel_padded = tf.pad(
                    self.kernel[:, j, i],
                    [[0, fft_size - self.kernel_size]]
                )

                # Apply FFT to kernel
                kernel_fft = tf.signal.rfft(kernel_padded)

                # Multiply in frequency domain
                conv_fft = inputs_fft[:, :, j] * kernel_fft

                # Apply IFFT
                conv_result = tf.signal.irfft(conv_fft)
                filter_outputs.append(conv_result[:, :seq_len])

            # Sum across input channels
            filter_output = tf.reduce_sum(tf.stack(filter_outputs, axis=-1), axis=-1)
            outputs.append(filter_output)

        # Stack filters
        output = tf.stack(outputs, axis=-1)

        # Add bias
        output = output + self.bias

        return output


class MultiTimeFrameCNN:
    """
    Multi-timeframe CNN model for financial prediction
    """

    def __init__(self, input_shape_5min, input_shape_1min, use_fft=False):
        self.input_shape_5min = input_shape_5min
        self.input_shape_1min = input_shape_1min
        self.use_fft = use_fft
        self.model = None
        self.fft_times = []
        self.standard_times = []

    def build_model(self):
        """Build the two-path CNN model"""
        # 5-minute path
        input_5min = layers.Input(shape=self.input_shape_5min, name='input_5min')

        if self.use_fft:
            x1 = FFTConv1D(filters=64, kernel_size=3, name='fft_conv1_5min')(input_5min)
        else:
            x1 = layers.Conv1D(filters=64, kernel_size=3, padding='same',
                               name='conv1_5min')(input_5min)

        x1 = layers.ReLU()(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(pool_size=2)(x1)

        if self.use_fft:
            x1 = FFTConv1D(filters=128, kernel_size=3, name='fft_conv2_5min')(x1)
        else:
            x1 = layers.Conv1D(filters=128, kernel_size=3, padding='same',
                               name='conv2_5min')(x1)

        x1 = layers.ReLU()(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Flatten()(x1)
        x1 = layers.Dense(64, activation='relu')(x1)
        x1 = layers.Dropout(0.3)(x1)

        # 1-minute path
        input_1min = layers.Input(shape=self.input_shape_1min, name='input_1min')

        if self.use_fft:
            x2 = FFTConv1D(filters=64, kernel_size=3, name='fft_conv1_1min')(input_1min)
        else:
            x2 = layers.Conv1D(filters=64, kernel_size=3, padding='same',
                               name='conv1_1min')(input_1min)

        x2 = layers.ReLU()(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling1D(pool_size=2)(x2)

        if self.use_fft:
            x2 = FFTConv1D(filters=128, kernel_size=3, name='fft_conv2_1min')(x2)
        else:
            x2 = layers.Conv1D(filters=128, kernel_size=3, padding='same',
                               name='conv2_1min')(x2)

        x2 = layers.ReLU()(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.Dropout(0.3)(x2)

        # Concatenate both paths
        concatenated = layers.Concatenate()([x1, x2])

        # Final layers
        x = layers.Dense(128, activation='relu')(concatenated)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(2, activation='softmax', name='output')(x)

        # Create model
        self.model = keras.Model(
            inputs=[input_5min, input_1min],
            outputs=output,
            name=f'MultiTimeFrame_{"FFT" if self.use_fft else "Standard"}'
        )

        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_5min_train, X_1min_train, y_train,
              X_5min_val, X_1min_val, y_val,
              epochs=50, batch_size=32):
        """Train the model"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        history = self.model.fit(
            [X_5min_train, X_1min_train],
            y_train,
            validation_data=([X_5min_val, X_1min_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def benchmark_inference(self, X_5min, X_1min, num_runs=100):
        """Benchmark inference time"""
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model.predict([X_5min[:1], X_1min[:1]], verbose=0)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return np.mean(times), np.std(times)


class FFTAnalyzer:
    """
    Analyze FFT vs standard convolution efficiency
    """

    @staticmethod
    def compare_convolution_methods(input_size, kernel_size, num_filters):
        """Compare time complexity of different convolution methods"""
        # Time domain complexity
        time_domain_ops = input_size * kernel_size * num_filters

        # Frequency domain complexity
        fft_size = 2 ** int(np.ceil(np.log2(input_size + kernel_size - 1)))
        fft_forward_ops = fft_size * np.log2(fft_size)
        fft_multiply_ops = fft_size * num_filters
        fft_backward_ops = fft_size * np.log2(fft_size) * num_filters
        freq_domain_ops = fft_forward_ops + fft_multiply_ops + fft_backward_ops

        return {
            'input_size': input_size,
            'kernel_size': kernel_size,
            'num_filters': num_filters,
            'time_domain_ops': time_domain_ops,
            'freq_domain_ops': freq_domain_ops,
            'speedup_factor': time_domain_ops / freq_domain_ops
        }

    @staticmethod
    def plot_complexity_comparison():
        """Plot complexity comparison for different input sizes"""
        input_sizes = [16, 32, 64, 128, 256, 512, 1024]
        kernel_size = 3
        num_filters = 64

        time_domain = []
        freq_domain = []

        for size in input_sizes:
            result = FFTAnalyzer.compare_convolution_methods(
                size, kernel_size, num_filters
            )
            time_domain.append(result['time_domain_ops'])
            freq_domain.append(result['freq_domain_ops'])

        plt.figure(figsize=(10, 6))
        plt.plot(input_sizes, time_domain, 'b-o', label='Time Domain', linewidth=2)
        plt.plot(input_sizes, freq_domain, 'r-s', label='Frequency Domain (FFT)', linewidth=2)
        plt.xlabel('Input Size')
        plt.ylabel('Number of Operations')
        plt.title('Convolution Complexity: Time Domain vs Frequency Domain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('fft_complexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.pause(10)
        plt.show()

        print("Complexity comparison plot saved!")


# Example usage
if __name__ == "__main__":
    # Load data
    X_5min_train = np.load('X_5min_train_eurusd.npy')
    X_1min_train = np.load('X_1min_train_eurusd.npy')
    y_train = np.load('y_train_eurusd.npy')

    # Split train/validation
    split_idx = int(0.9 * len(y_train))
    X_5min_val = X_5min_train[split_idx:]
    X_1min_val = X_1min_train[split_idx:]
    y_val = y_train[split_idx:]

    X_5min_train = X_5min_train[:split_idx]
    X_1min_train = X_1min_train[:split_idx]
    y_train = y_train[:split_idx]

    print("Training standard CNN model...")
    model_standard = MultiTimeFrameCNN(
        input_shape_5min=X_5min_train.shape[1:],
        input_shape_1min=X_1min_train.shape[1:],
        use_fft=False
    )
    model_standard.build_model()
    model_standard.compile_model()

    history_standard = model_standard.train(
        X_5min_train, X_1min_train, y_train,
        X_5min_val, X_1min_val, y_val,
        epochs=50, batch_size=32
    )

    # Save model
    model_standard.model.save('model_standard.keras')

    # Benchmark
    mean_time, std_time = model_standard.benchmark_inference(X_5min_val, X_1min_val)
    print(f"Standard model inference time: {mean_time:.3f} ± {std_time:.3f} ms")

    # FFT complexity analysis
    print("\nFFT Complexity Analysis:")
    analyzer = FFTAnalyzer()
    analyzer.plot_complexity_comparison()

    for size in [32, 64, 128, 256]:
        result = analyzer.compare_convolution_methods(size, 3, 64)
        print(f"\nInput size: {result['input_size']}")
        print(f"Time domain ops: {result['time_domain_ops']:,}")
        print(f"Freq domain ops: {result['freq_domain_ops']:,.0f}")
        print(f"Speedup factor: {result['speedup_factor']:.2f}x")
