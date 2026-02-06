#!/usr/bin/env python3
"""
Evaluate a trained federated QCNN model.
Generates comprehensive results: accuracy, loss, classification report, confusion matrix.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create output directory for evaluation results
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def evaluate_model(model_path, X_test, y_test, class_names=None):
    """
    Evaluate model and generate comprehensive results.

    Args:
        model_path: Path to trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names

    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 70)
    print("FEDERATED QCNN MODEL EVALUATION")
    print("=" * 70)

    # Load model (handle Keras version compatibility)
    print(f"\nLoading model from: {model_path}")

    import warnings

    warnings.filterwarnings("ignore")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative method...")

        # Alternative: Load weights only
        try:
            from model import build_qcnn_model

            model = build_qcnn_model()
            model.load_weights(model_path)
            print("Model weights loaded successfully!")
        except Exception as e2:
            print(f"Weight loading also failed: {e2}")
            raise Exception("Could not load model. Check TensorFlow version.")

    print(f"Model loaded successfully!")

    # Get model summary
    print("\n" + "-" * 70)
    print("MODEL ARCHITECTURE")
    print("-" * 70)
    model.summary()

    # Count parameters
    total_params = model.count_params()
    trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("EVALUATION RESULTS")
    print("-" * 70)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n[OK] Test Loss: {loss:.4f}")
    print(f"[OK] Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_pred_proba = model.predict(X_test)

    # Classification Report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # Confusion Matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")
    print("\nConfusion Matrix:")
    print(cm)

    # Per-class metrics
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)

    per_class_metrics = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy_i = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0

        per_class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy_i,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

        print(f"\n{cls}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy_i:.4f}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Overall metrics
    print("\n" + "-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)

    macro_precision = np.mean(
        [per_class_metrics[cls]["precision"] for cls in class_names]
    )
    macro_recall = np.mean([per_class_metrics[cls]["recall"] for cls in class_names])
    macro_f1 = np.mean([per_class_metrics[cls]["f1_score"] for cls in class_names])

    weighted_precision = np.sum(
        [
            per_class_metrics[cls]["precision"] * np.sum(y_test == i)
            for i, cls in enumerate(class_names)
        ]
    ) / len(y_test)
    weighted_recall = np.sum(
        [
            per_class_metrics[cls]["recall"] * np.sum(y_test == i)
            for i, cls in enumerate(class_names)
        ]
    ) / len(y_test)
    weighted_f1 = np.sum(
        [
            per_class_metrics[cls]["f1_score"] * np.sum(y_test == i)
            for i, cls in enumerate(class_names)
        ]
    ) / len(y_test)

    print(f"\nMacro Averages:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")

    print(f"\nWeighted Averages:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall: {weighted_recall:.4f}")
    print(f"  F1-Score: {weighted_f1:.4f}")

    # Dataset info
    print("\n" + "-" * 70)
    print("DATASET INFORMATION")
    print("-" * 70)

    print(f"Test samples: {len(y_test)}")
    print(f"Feature dimensions: {X_test.shape[1]}")
    print(f"Number of classes: {len(class_names)}")

    class_dist = np.bincount(y_test)
    print(f"\nClass distribution in test set:")
    for i, (cls, count) in enumerate(zip(class_names, class_dist)):
        print(f"  {cls}: {count} samples ({count / len(y_test) * 100:.1f}%)")

    # Results dictionary
    results = {
        "model_path": model_path,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "test_samples": len(y_test),
        "feature_dimensions": X_test.shape[1],
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_distribution": {
            cls: int(count) for cls, count in zip(class_names, class_dist)
        },
        "per_class_metrics": per_class_metrics,
        "macro_metrics": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1),
        },
        "weighted_metrics": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1),
        },
        "confusion_matrix": cm.tolist(),
    }

    return results, model, y_pred, y_pred_proba


def plot_results(results, y_test, y_pred, y_pred_proba, output_dir="."):
    """Generate and save result plots."""
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)

    class_names = results["class_names"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cm = np.array(results["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Federated QCNN", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/confusion_matrix_{timestamp}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[OK] Saved: confusion_matrix_{timestamp}.png")

    # 2. Per-Class Metrics Bar Chart
    plt.figure(figsize=(12, 6))
    metrics = ["precision", "recall", "f1_score"]
    x = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [results["per_class_metrics"][cls][metric] for cls in class_names]
        plt.bar(x + i * width, values, width, label=metric.capitalize())

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Per-Class Precision, Recall, and F1-Score", fontsize=14)
    plt.xticks(x + width, class_names, rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/per_class_metrics_{timestamp}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[OK] Saved: per_class_metrics_{timestamp}.png")

    # 3. Accuracy Distribution
    plt.figure(figsize=(8, 6))
    accuracies = [results["per_class_metrics"][cls]["accuracy"] for cls in class_names]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(class_names)))
    bars = plt.bar(class_names, accuracies, color=colors)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Per-Class Accuracy", fontsize=14)
    plt.ylim(0, 1.2)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/accuracy_by_class_{timestamp}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[OK] Saved: accuracy_by_class_{timestamp}.png")

    # 4. Prediction Probability Distribution
    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(class_names):
        class_probs = y_pred_proba[y_test == i, i]
        plt.hist(class_probs, bins=20, alpha=0.5, label=cls)

    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Predicted Probabilities by True Class", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/probability_distribution_{timestamp}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[OK] Saved: probability_distribution_{timestamp}.png")

    # 5. Summary Statistics
    plt.figure(figsize=(10, 6))

    # Create summary text
    summary_text = f"""
    FEDERATED QCNN EVALUATION SUMMARY
    =================================
    
    Model: {os.path.basename(results["model_path"])}
    Evaluation Time: {results["evaluation_time"]}
    
    DATASET
    -------
    Test Samples: {results["test_samples"]}
    Feature Dimensions: {results["feature_dimensions"]}
    Number of Classes: {results["num_classes"]}
    
    PERFORMANCE
    -----------
    Test Loss: {results["test_loss"]:.4f}
    Test Accuracy: {results["test_accuracy"]:.4f} ({results["test_accuracy"] * 100:.2f}%)
    
    MACRO AVERAGES
    --------------
    Precision: {results["macro_metrics"]["precision"]:.4f}
    Recall: {results["macro_metrics"]["recall"]:.4f}
    F1-Score: {results["macro_metrics"]["f1_score"]:.4f}
    
    WEIGHTED AVERAGES
    -----------------
    Precision: {results["weighted_metrics"]["precision"]:.4f}
    Recall: {results["weighted_metrics"]["recall"]:.4f}
    F1-Score: {results["weighted_metrics"]["f1_score"]:.4f}
    
    MODEL SIZE
    ----------
    Total Parameters: {results["total_parameters"]:,}
    Trainable Parameters: {results["trainable_parameters"]:,}
    """

    plt.text(
        0.05,
        0.95,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.axis("off")
    plt.title("Evaluation Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/evaluation_summary_{timestamp}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[OK] Saved: evaluation_summary_{timestamp}.png")

    print(f"\n All visualizations saved to: {output_dir}")


def save_results_report(results, output_path="evaluation_report.txt"):
    """Save detailed results to a text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FEDERATED QCNN MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Evaluation Time: {results['evaluation_time']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write(
            f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy'] * 100:.2f}%)\n\n"
        )

        f.write("-" * 70 + "\n")
        f.write("MACRO AVERAGES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Precision: {results['macro_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {results['macro_metrics']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['macro_metrics']['f1_score']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("WEIGHTED AVERAGES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Precision: {results['weighted_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {results['weighted_metrics']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['weighted_metrics']['f1_score']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n")
        for cls, metrics in results["per_class_metrics"].items():
            f.write(f"\n{cls}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        cm = np.array(results["confusion_matrix"])
        f.write(str(cm) + "\n\n")

        f.write("-" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        for cls, count in results["class_distribution"].items():
            f.write(f"{cls}: {count} samples\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[OK] Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained federated QCNN model"
    )
    parser.add_argument("--model", type=str, help="Path to trained model (.h5 file)")
    parser.add_argument(
        "--features", type=str, required=True, help="Path to test features (.npy file)"
    )
    parser.add_argument(
        "--labels", type=str, required=True, help="Path to test labels (.npy file)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="MildDemented,ModerateDemented,NonDemented,VeryMildDemented",
        help="Comma-separated class names",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EVAL_OUTPUT_DIR,
        help="Output directory for plots and report",
    )

    args = parser.parse_args()

    # Load test data
    print("Loading test data...")
    X_test = np.load(args.features)
    y_test = np.load(args.labels)
    print(f"Loaded {len(y_test)} test samples")

    # Parse class names
    class_names = [c.strip() for c in args.classes.split(",")]

    # Find model if not specified
    if args.model is None:
        import glob

        # Look in logs_and_models folder
        logs_dir = os.path.join(os.path.dirname(__file__), "logs_and_models")
        models = glob.glob(os.path.join(logs_dir, "global_model_fednova_*.h5"))
        if models:
            args.model = max(models, key=os.path.getmtime)
            print(f"Using latest model: {args.model}")
        else:
            # Try current directory
            models = glob.glob("global_model_fednova_*.h5")
            if models:
                args.model = max(models, key=os.path.getmtime)
                print(f"Using latest model: {args.model}")
            else:
                print("[FAIL] No model found! Please specify --model")
                sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Evaluate model
    results, model, y_pred, y_pred_proba = evaluate_model(
        args.model, X_test, y_test, class_names
    )

    # Generate plots
    plot_results(results, y_test, y_pred, y_pred_proba, args.output)

    # Save report
    report_path = os.path.join(args.output, "evaluation_report.txt")
    save_results_report(results, report_path)

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n Key Results:")
    print(f"   Accuracy: {results['test_accuracy'] * 100:.2f}%")
    print(f"   Loss: {results['test_loss']:.4f}")
    print(f"   F1-Score (Macro): {results['macro_metrics']['f1_score']:.4f}")
    print(f"\n Output Files saved to: {args.output}")
    print(f"   - Confusion Matrix")
    print(f"   - Per-Class Metrics")
    print(f"   - Accuracy by Class")
    print(f"   - Probability Distribution")
    print(f"   - Evaluation Summary")
    print(f"   - Detailed Report: evaluation_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
