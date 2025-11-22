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
            'predictions': y_pred
        }

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-Score: {f1_score * 100:.2f}%")

        return accuracy

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

    evaluator = ModelEvaluator()

    # 1. Train and evaluate MLP
    print("\n" + "=" * 80)
    print("Training MLP...")
    mlp = BaselineModels.build_mlp(X_5min_train.shape[1:])
    mlp.fit(X_5min_train, y_train, epochs=30, batch_size=32,
            validation_split=0.1, verbose=1)
    evaluator.evaluate_model(mlp, X_5min_test, y_test, 'MLP')

    # 2. Train and evaluate LSTM
    print("\n" + "=" * 80)
    print("Training LSTM...")
    lstm = BaselineModels.build_lstm(X_5min_train.shape[1:])
    lstm.fit(X_5min_train, y_train, epochs=30, batch_size=32,
             validation_split=0.1, verbose=1)
    evaluator.evaluate_model(lstm, X_5min_test, y_test, 'LSTM')

    # 3. Train and evaluate 1D-CNN
    print("\n" + "=" * 80)
    print("Training 1D-CNN...")
    cnn_1d = BaselineModels.build_1d_cnn(X_5min_train.shape[1:])
    cnn_1d.fit(X_5min_train, y_train, epochs=30, batch_size=32,
               validation_split=0.1, verbose=1)
    evaluator.evaluate_model(cnn_1d, X_5min_test, y_test, '1D-CNN')

    # 4. Train and evaluate Hybrid CNN-LSTM
    print("\n" + "=" * 80)
    print("Training Hybrid CNN-LSTM...")
    hybrid = BaselineModels.build_hybrid_cnn_lstm(X_5min_train.shape[1:])
    hybrid.fit(X_5min_train, y_train, epochs=30, batch_size=32,
               validation_split=0.1, verbose=1)
    evaluator.evaluate_model(hybrid, X_5min_test, y_test, 'Hybrid CNN-LSTM')

    # 5. Evaluate proposed model
    print("\n" + "=" * 80)
    print("Evaluating Proposed Multi-Timeframe Model...")
    proposed_model = keras.models.load_model('model_standard.h5')
    evaluator.evaluate_model(proposed_model, X_5min_test, y_test,
                             'Proposed Multi-Timeframe CNN',
                             X_test_2=X_1min_test, is_dual_input=True)

    # Generate visualizations and reports
    evaluator.plot_confusion_matrices()
    evaluator.plot_comparison()
    evaluator.generate_report()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)