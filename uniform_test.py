"""
Test Notebook Code - Uniform Distribution Test with FIXES
Copy cells into Jupyter notebook

PROBLEM IDENTIFIED: Model predicts ALL samples as NonDemented (class 2)
FIXES APPLIED:
1. Remove initial BatchNormalization (before any training)
2. Add class weights to handle any imbalance
3. Lower learning rate
4. Add RandomForest sanity check first
5. Try simpler architecture
"""

# =============================================================================
# CELL 1: Imports
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Imports loaded successfully")


# =============================================================================
# CELL 2: Load ALL Quantum Features from All Clients
# =============================================================================
BASE_PATH = "quantum_features"

all_features = []
all_labels = []

for client_id in range(1, 6):
    features_path = os.path.join(BASE_PATH, f"client_{client_id}", "train_features.npy")
    labels_path = os.path.join(BASE_PATH, f"client_{client_id}", "train_labels.npy")

    features = np.load(features_path)
    labels = np.load(labels_path)

    all_features.append(features)
    all_labels.append(labels)

    print(f"Client {client_id}: {len(features)} samples")

X_all = np.vstack(all_features)
y_all = np.concatenate(all_labels)

print(f"\nTotal combined samples: {len(X_all)}")
print(f"Feature shape: {X_all.shape}")


# =============================================================================
# CELL 3: Create Uniform IID Split AND Compute Class Weights
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=None, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Compute class weights (to handle any imbalance)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"\nClass weights: {class_weight_dict}")


# =============================================================================
# CELL 4: SANITY CHECK - RandomForest on Quantum Features
# =============================================================================
print("=" * 60)
print("SANITY CHECK: RandomForest on quantum features")
print("=" * 60)

# Use a portion for quick test
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nRandomForest Accuracy: {rf_acc * 100:.2f}%")
print("\nRandomForest Classification Report:")
print(classification_report(y_test, rf_pred))

if rf_acc < 0.30:
    print("\n*** WARNING: RandomForest also performs poorly! ***")
    print("This suggests the quantum features may not be distinguishable.")
else:
    print(
        f"\n*** RandomForest achieves {rf_acc * 100:.2f}% - Features ARE separable! ***"
    )


# =============================================================================
# CELL 5: Build FIXED QCNN Model
# =============================================================================
INPUT_DIM = 4096
NUM_CLASSES = 4
LEARNING_RATE = 0.0001  # Much lower learning rate
DENSE_LAYERS = [512, 256, 128]  # Simpler architecture


def build_qcnn_model_v2():
    model = K.Sequential(
        [
            layers.Input(shape=(INPUT_DIM,)),
            layers.Flatten(),
            # First dense layer WITHOUT BatchNorm at input
            layers.Dense(DENSE_LAYERS[0], kernel_initializer="he_uniform"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.4),  # Higher dropout
            layers.Dense(DENSE_LAYERS[1], kernel_initializer="he_uniform"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(DENSE_LAYERS[2], kernel_initializer="he_uniform"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


model = build_qcnn_model_v2()
model.summary()


# =============================================================================
# CELL 6: Train with Class Weights and Callbacks
# =============================================================================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

EPOCHS = 50
BATCH_SIZE = 32

print("Training QCNN model with class weights...")
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,  # IMPORTANT: Use class weights
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# =============================================================================
# CELL 7: Evaluate Model
# =============================================================================
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'=' * 50}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"{'=' * 50}")

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
print(classification_report(y_test, y_pred_classes, target_names=class_names))

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    mask = y_test == i
    class_acc = np.mean(y_pred_classes[mask] == i)
    print(f"  {name}: {class_acc * 100:.2f}%")


# =============================================================================
# CELL 8: Confusion Matrix
# =============================================================================
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# =============================================================================
# CELL 9: Training Curves
# =============================================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()


# =============================================================================
# CELL 10: Try Even Simpler Model (Diagnostic)
# =============================================================================
print("=" * 60)
print("DIAGNOSTIC: Try simpler model (256-128 output)")
print("=" * 60)


def build_simple_model():
    model = K.Sequential(
        [
            layers.Input(shape=(INPUT_DIM,)),
            layers.Flatten(),
            layers.Dense(256, kernel_initializer="he_uniform"),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Dense(128, kernel_initializer="he_uniform"),
            layers.ReLU(),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


simple_model = build_simple_model()

history_simple = simple_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1,
)

# Evaluate
simple_pred = simple_model.predict(X_test, verbose=0)
simple_pred_classes = np.argmax(simple_pred, axis=1)
simple_acc = accuracy_score(y_test, simple_pred_classes)

print(f"\nSimple Model Test Accuracy: {simple_acc * 100:.2f}%")
print(classification_report(y_test, simple_pred_classes, target_names=class_names))


# =============================================================================
# CELL 11: Global Test Set Evaluation
# =============================================================================
global_test_features_path = os.path.join(BASE_PATH, "global_test_features.npy")
global_test_labels_path = os.path.join(BASE_PATH, "global_test_labels.npy")

if os.path.exists(global_test_features_path) and os.path.exists(
    global_test_labels_path
):
    X_global_test = np.load(global_test_features_path)
    y_global_test = np.load(global_test_labels_path)

    print(f"\nGlobal test set: {len(X_global_test)} samples")

    # Use best model (either the main one or simple one based on val accuracy)
    if simple_acc > test_accuracy:
        print("Using Simple Model for global test")
        best_model = simple_model
    else:
        print("Using Main Model for global test")
        best_model = model

    global_loss, global_acc = best_model.evaluate(
        X_global_test, y_global_test, verbose=0
    )
    print(f"\n{'=' * 50}")
    print(f"Global Test Accuracy: {global_acc * 100:.2f}%")
    print(f"Global Test Loss: {global_loss:.4f}")
    print(f"{'=' * 50}")

    y_global_pred = best_model.predict(X_global_test, verbose=0)
    y_global_pred_classes = np.argmax(y_global_pred, axis=1)

    print("\nClassification Report (Global Test):")
    print(
        classification_report(
            y_global_test, y_global_pred_classes, target_names=class_names
        )
    )


# =============================================================================
# CELL 12: Feature Analysis - Check if features differ between classes
# =============================================================================
print("=" * 60)
print("FEATURE ANALYSIS: Mean feature values per class")
print("=" * 60)

for cls in range(NUM_CLASSES):
    mask = y_all == cls
    mean_features = X_all[mask].mean(axis=0)
    std_features = X_all[mask].std(axis=0)
    print(f"Class {cls} ({class_names[cls]}):")
    print(f"  Mean: {mean_features.mean():.4f}, Std: {std_features.mean():.4f}")
    print(f"  Min: {mean_features.min():.4f}, Max: {mean_features.max():.4f}")

# Check distance between class means
print("\nDistance between class centers (cosine similarity):")
from sklearn.metrics.pairwise import cosine_similarity

centers = []
for cls in range(NUM_CLASSES):
    mask = y_all == cls
    centers.append(X_all[mask].mean(axis=0))
centers = np.array(centers)

sim_matrix = cosine_similarity(centers)
print(sim_matrix)
