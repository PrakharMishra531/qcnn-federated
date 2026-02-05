import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import NUM_CLASSES


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum([np.prod(layer.shape) for layer in model.trainable_variables])


def get_class_distribution(labels):
    """Get class distribution from labels."""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def calculate_flops(model):
    """Estimate FLOPs for model (simplified)."""
    total_flops = 0
    for layer in model.layers:
        if hasattr(layer, "output_shape"):
            output_dim = np.prod(layer.output_shape)
            if hasattr(layer, "weights"):
                weights_dim = np.prod(layer.weights[0].shape) if layer.weights else 0
                total_flops += weights_dim * output_dim
    return total_flops


def evaluate_model(model, X, y, class_names=None):
    """Evaluate model and return metrics."""
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]

    loss, accuracy = model.evaluate(X, y, verbose=0)
    y_pred = np.argmax(model.predict(X), axis=1)

    report = classification_report(
        y, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y, y_pred)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def plot_training_history(history, title="Training History"):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss plot
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def log_metrics(logger, round_num, metrics):
    """Log metrics to logger."""
    logger.info(f"Round {round_num}:")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")


def measure_time(func):
    """Decorator to measure function execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def normalize_weights(weights):
    """Normalize weight vectors."""
    total = sum(np.linalg.norm(w) for w in weights)
    if total == 0:
        return weights
    return [w / total for w in weights]


def clip_weights(weights, max_norm=1.0):
    """Clip weights to maximum norm."""
    norm = np.linalg.norm([np.linalg.norm(w) for w in weights])
    if norm > max_norm:
        return [w * max_norm / norm for w in weights]
    return weights


if __name__ == "__main__":
    print("Utility functions loaded successfully")
