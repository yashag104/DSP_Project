import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BaselineModels:
    """Baseline models for comparison"""

    @staticmethod
    def build_mlp(input_shape):
        """Multi-Layer Perceptron"""
        model = keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')
        ], name='MLP')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def build_lstm(input_shape):
        """LSTM model"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')
        ], name='LSTM')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def build_1d_cnn(input_shape):
        """Single timeframe 1D-CNN"""
        model = keras.Sequential([
            layers.Conv1D(64, 3, padding='same', input_shape=input_shape),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, padding='same'),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ], name='1D_CNN')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def build_hybrid_cnn_lstm(input_shape):
        """Hybrid CNN-LSTM model"""
        model = keras.Sequential([
            layers.Conv1D(64, 3, padding='same', input_shape=input_shape),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')
        ], name='Hybrid_CNN_LSTM')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self):
        self.results = {}
        self.training_histories = {}

    def evaluate_model(self, model, X_test, y_test, model_name,
                       X_test_2=None, is_dual_input=False):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")

        # Predict
        if is_dual_input and X_test_2 is not None:
            y_pred_proba = model.predict([X_test, X_test_2])
        else:
            y_pred_proba = model.predict(X_test)

        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        self.results[model_name] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-Score: {f1_score * 100:.2f}%")

        return accuracy

    def plot_lstm_actual_price_prediction(self, lstm_model, X_test_scaled, y_test,
                                          X_test_raw,
                                          save_path='LSTM_actual_price_prediction.png',
                                          num_samples=500):
        """
        Plot ACTUAL prices vs LSTM predicted direction
        Shows real forex prices with predicted vs actual trends

        Parameters:
        -----------
        lstm_model: trained LSTM model
        X_test_scaled: scaled test data (for model prediction)
        y_test: actual labels (0=down, 1=up)
        X_test_raw: RAW UNSCALED test data with actual prices
        save_path: path to save the plot
        num_samples: number of samples to plot
        """
        print("\n" + "=" * 80)
        print("Generating LSTM Actual Price Prediction Plot...")
        print("=" * 80)

        # Get LSTM predictions
        y_pred_proba = lstm_model.predict(X_test_scaled)
        y_pred_class = np.argmax(y_pred_proba, axis=1)

        # Extract actual Close prices from RAW UNSCALED data
        if X_test_raw.min() < 0:
            raise ValueError("ERROR: X_test_raw contains scaled data (negative values)! "
                             "You must pass the ORIGINAL UNSCALED data. "
                             "Save it during preprocessing before scaling: "
                             "np.save('X_5min_test_gbpusd_raw.npy', X_test_BEFORE_SCALING)")

        current_prices = X_test_raw[:, -1, 3]  # Last timestep Close price

        # Get next actual prices (ground truth)
        actual_next_prices = np.zeros(len(current_prices))
        for i in range(len(current_prices) - 1):
            actual_next_prices[i] = X_test_raw[i + 1, -1, 3]
        actual_next_prices[-1] = current_prices[-1]

        # Limit samples
        plot_length = min(num_samples, len(current_prices))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        x_axis = np.arange(plot_length)

        # ===== Plot 1: Current vs Next Actual Prices =====
        ax1.plot(x_axis, current_prices[:plot_length], 'b-', linewidth=2,
                 label='Current Price', alpha=0.8)
        ax1.plot(x_axis, actual_next_prices[:plot_length], 'g-', linewidth=2,
                 label='Actual Next Price', alpha=0.8)

        ax1.set_ylabel('Price (Original)', fontsize=13, fontweight='bold')
        ax1.set_title('Current Price vs Actual Next Price (Real Unscaled Values)',
                      fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Add price statistics
        price_min = np.min(current_prices[:plot_length])
        price_max = np.max(current_prices[:plot_length])
        price_mean = np.mean(current_prices[:plot_length])
        ax1.text(0.02, 0.95, f'Min: {price_min:.5f}\nMax: {price_max:.5f}\nMean: {price_mean:.5f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== Plot 2: Prediction Accuracy on Price Movement =====
        for i in range(plot_length):
            if y_pred_class[i] == y_test[i]:  # Correct prediction
                ax2.plot([x_axis[i], x_axis[i]],
                         [current_prices[i], actual_next_prices[i]],
                         'g-', linewidth=1.5, alpha=0.6)
            else:  # Wrong prediction
                ax2.plot([x_axis[i], x_axis[i]],
                         [current_prices[i], actual_next_prices[i]],
                         'r-', linewidth=1.5, alpha=0.6)

        ax2.plot(x_axis, current_prices[:plot_length], 'b-', linewidth=2,
                 label='Current Price', alpha=0.8)
        ax2.plot(x_axis, actual_next_prices[:plot_length], 'k--', linewidth=1.5,
                 label='Actual Next Price', alpha=0.5)

        ax2.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Price (Original)', fontsize=13, fontweight='bold')
        ax2.set_title('LSTM Prediction Accuracy (Green=Correct Direction, Red=Wrong Direction)',
                      fontsize=15, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('LSTM Price Prediction - Actual Unscaled Prices',
                     fontsize=17, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
        plt.close()

        # Calculate accuracy
        direction_accuracy = np.sum(y_pred_class[:plot_length] == y_test[:plot_length]) / plot_length * 100
        print(f"Direction Prediction Accuracy: {direction_accuracy:.2f}%")

        return {
            'current_prices': current_prices[:plot_length],
            'actual_next_prices': actual_next_prices[:plot_length],
            'direction_accuracy': direction_accuracy
        }

    def plot_training_history(self, history, model_name, save_path=None):
        """Plot training history for a model"""
        if save_path is None:
            save_path = f'{model_name}_training_history.png'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()

    def plot_confusion_matrices(self, save_path='confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        axes = axes.ravel()

        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.2f}%')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        # Hide extra subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to {save_path}")
        plt.close()

    def plot_comparison(self, save_path='model_comparison.png'):
        """Plot comparison of all models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())

        data = {metric: [] for metric in metrics}
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(self.results[model_name][metric])

        # Create comparison plot
        x = np.arange(len(model_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, metric in enumerate(metrics):
            offset = width * (idx - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize())

        ax.set_xlabel('Models')
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.close()

    def print_comparison_table(self):
        print("\n" + "=" * 120)
        print("MODEL COMPARISON TABLE")
        print("=" * 120)

        # Create DataFrame
        df_data = []
        for model_name, results in self.results.items():
            df_data.append({
                'Model': model_name,
                'Accuracy (%)': f"{results['accuracy']:.2f}",
                'Precision (%)': f"{results['precision']:.2f}",
                'Recall (%)': f"{results['recall']:.2f}",
                'F1-Score (%)': f"{results['f1_score']:.2f}",
                'TP': results['tp'],
                'TN': results['tn'],
                'FP': results['fp'],
                'FN': results['fn']
            })

        df = pd.DataFrame(df_data)
        df = df.sort_values('Accuracy (%)', ascending=False)

        print(df.to_string(index=False))
        print("=" * 120)

        # Save to CSV
        df.to_csv('model_comparison_metrics.csv', index=False)
        print("\nComparison metrics saved to 'model_comparison_metrics.csv'")

    def generate_report(self, save_path='evaluation_report.txt'):
        """Generate detailed evaluation report"""
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Sort by accuracy
            sorted_results = sorted(self.results.items(),
                                    key=lambda x: x[1]['accuracy'],
                                    reverse=True)

            for rank, (model_name, results) in enumerate(sorted_results, 1):
                f.write(f"\n{rank}. {model_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Accuracy:  {results['accuracy']:.2f}%\n")
                f.write(f"Precision: {results['precision']:.2f}%\n")
                f.write(f"Recall:    {results['recall']:.2f}%\n")
                f.write(f"F1-Score:  {results['f1_score']:.2f}%\n")
                f.write(f"\nConfusion Matrix:\n{results['confusion_matrix']}\n")
                f.write(f"True Positives:  {results['tp']}\n")
                f.write(f"True Negatives:  {results['tn']}\n")
                f.write(f"False Positives: {results['fp']}\n")
                f.write(f"False Negatives: {results['fn']}\n")

            # Best model
            best_model = sorted_results[0][0]
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"BEST MODEL: {best_model}\n")
            f.write(f"Accuracy: {self.results[best_model]['accuracy']:.2f}%\n")
            f.write("=" * 80 + "\n")

        print(f"\nDetailed report saved to {save_path}")


# Main evaluation script
if __name__ == "__main__":
    # Load training data (EUR/USD)
    X_5min_train = np.load('X_5min_train_eurusd.npy')
    X_1min_train = np.load('X_1min_train_eurusd.npy')
    y_train = np.load('y_train_eurusd.npy')

    # Load testing data (GBP/USD - Cross-currency testing)
    X_5min_test = np.load('X_5min_test_gbpusd.npy')
    X_1min_test = np.load('X_1min_test_gbpusd.npy')
    y_test = np.load('y_test_gbpusd.npy')

    # Load raw test data (unscaled) for price visualization
    try:
        X_5min_test_raw = np.load('X_5min_test_gbpusd_raw.npy')
        print("✓ Raw test data loaded successfully!")
    except FileNotFoundError:
        print("WARNING: Raw test data not found!")
        X_5min_test_raw = X_5min_test

    evaluator = ModelEvaluator()

    # 1. Train and evaluate MLP
    print("\n" + "=" * 80)
    print("Training MLP...")
    mlp = BaselineModels.build_mlp(X_5min_train.shape[1:])
    history_mlp = mlp.fit(X_5min_train, y_train, epochs=30, batch_size=32,
                          validation_split=0.1, verbose=1)
    evaluator.plot_training_history(history_mlp, 'MLP')
    evaluator.evaluate_model(mlp, X_5min_test, y_test, 'MLP')

    # 2. Train and evaluate LSTM
    print("\n" + "=" * 80)
    print("Training LSTM...")
    lstm = BaselineModels.build_lstm(X_5min_train.shape[1:])
    history_lstm = lstm.fit(X_5min_train, y_train, epochs=30, batch_size=32,
                            validation_split=0.1, verbose=1)
    evaluator.plot_training_history(history_lstm, 'LSTM')
    evaluator.evaluate_model(lstm, X_5min_test, y_test, 'LSTM')

    # Generate LSTM actual price prediction plot
    print("\n" + "=" * 80)
    print("Generating LSTM Actual Price Prediction Analysis...")
    print("=" * 80)

    lstm_price_stats = evaluator.plot_lstm_actual_price_prediction(
        lstm_model=lstm,
        X_test_scaled=X_5min_test,
        y_test=y_test,
        X_test_raw=X_5min_test_raw,
        save_path='LSTM_actual_price_prediction.png',
        num_samples=500
    )

    # 3. Train and evaluate 1D-CNN
    print("\n" + "=" * 80)
    print("Training 1D-CNN...")
    cnn_1d = BaselineModels.build_1d_cnn(X_5min_train.shape[1:])
    history_cnn = cnn_1d.fit(X_5min_train, y_train, epochs=30, batch_size=32,
                             validation_split=0.1, verbose=1)
    evaluator.plot_training_history(history_cnn, '1D-CNN')
    evaluator.evaluate_model(cnn_1d, X_5min_test, y_test, '1D-CNN')

    # 4. Train and evaluate Hybrid CNN-LSTM
    print("\n" + "=" * 80)
    print("Training Hybrid CNN-LSTM...")
    hybrid = BaselineModels.build_hybrid_cnn_lstm(X_5min_train.shape[1:])
    history_hybrid = hybrid.fit(X_5min_train, y_train, epochs=30, batch_size=32,
                                validation_split=0.1, verbose=1)
    evaluator.plot_training_history(history_hybrid, 'Hybrid_CNN_LSTM')
    evaluator.evaluate_model(hybrid, X_5min_test, y_test, 'Hybrid CNN-LSTM')

    # 5. Evaluate proposed model
    print("\n" + "=" * 80)
    print("Evaluating Proposed Multi-Timeframe Model...")
    try:
        proposed_model = keras.models.load_model('model_standard.h5')
        evaluator.evaluate_model(proposed_model, X_5min_test, y_test,
                                 'Proposed Multi-Timeframe CNN',
                                 X_test_2=X_1min_test, is_dual_input=True)
    except:
        print("WARNING: Proposed model not found. Skipping this evaluation.")

    # Generate visualizations and reports
    evaluator.plot_confusion_matrices()
    evaluator.plot_comparison()
    evaluator.print_comparison_table()
    evaluator.generate_report()