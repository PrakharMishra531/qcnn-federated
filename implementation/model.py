import os
import pickle

import numpy as np
import tensorflow as tf
from config import DENSE_LAYERS, DROPOUT_RATES, INPUT_DIM, LEARNING_RATE, NUM_CLASSES
from tensorflow import keras as K
from tensorflow.keras import layers

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# def setup_gpu():
#     """Enable GPU memory growth to prevent Out-of-Memory errors on Jetson Nano."""
#     gpus = tf.config.experimental.list_physical_devices("GPU")
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             print(f"GPU detected: {len(gpus)} GPU(s) available")
#         except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(f"GPU config error: {e}")
#     else:
#         print("No GPU detected. Running on CPU.")


# # Initialize GPU settings immediately
# setup_gpu()


def build_qcnn_model(input_dim=None, num_classes=NUM_CLASSES):
    """
    Build QCNN model - classical MLP for quantum feature classification.

    Args:
        input_dim: Input feature dimension (default from config)
        num_classes: Number of output classes (default 4)

    Returns:
        Compiled Keras model
    """
    if input_dim is None:
        input_dim = INPUT_DIM

    model = K.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Flatten(),
            layers.BatchNormalization(),
        ]
    )

    # Add dense layers with LeakyReLU, BatchNorm, and Dropout
    for i, units in enumerate(DENSE_LAYERS[:-1]):
        model.add(layers.Dense(units, kernel_initializer="he_uniform"))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.BatchNormalization())

        if i < len(DROPOUT_RATES):
            model.add(layers.Dropout(DROPOUT_RATES[i]))

    # Final output layer
    model.add(layers.Dense(DENSE_LAYERS[-1], kernel_initializer="he_uniform"))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile model
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_weights(model):
    """Extract model weights as numpy arrays."""
    weights = []
    for layer in model.layers:
        if len(layer.weights) > 0:
            layer_weights = []
            for w in layer.weights:
                if hasattr(w, "numpy"):
                    layer_weights.append(w.numpy())
                else:
                    layer_weights.append(np.array(w))
            weights.append(layer_weights)
    return weights


def get_flat_weights(model):
    """Extract model weights as a flat list of numpy arrays (for serialization)."""
    flat_weights = []
    for layer in model.layers:
        if len(layer.weights) > 0:
            for w in layer.weights:
                if hasattr(w, "numpy"):
                    flat_weights.append(w.numpy())
                else:
                    flat_weights.append(np.array(w))
    return flat_weights


def set_weights_from_flat(model, flat_weights):
    """Set model weights from a flat list of numpy arrays."""
    weight_idx = 0
    for layer in model.layers:
        if len(layer.weights) > 0:
            layer_weights = []
            for w in layer.weights:
                layer_weights.append(flat_weights[weight_idx])
                weight_idx += 1
            layer.set_weights(layer_weights)
    return model


def set_model_weights(model, weights):
    """Set model weights from numpy arrays."""
    for i, layer in enumerate(model.layers):
        if len(layer.weights) > 0:
            layer.set_weights(weights[i])
    return model


def compute_weight_delta(local_weights, global_weights):
    """Compute delta = local_weights - global_weights."""
    return [lw - gw for lw, gw in zip(local_weights, global_weights)]


def serialize_weights(weights):
    """Serialize weights in a NumPy-version-agnostic way using raw bytes.

    Instead of .tolist() (which inflates float32 arrays ~6x by converting to
    Python floats), we store raw bytes alongside dtype/shape metadata.
    pickle protocol 2 is used for maximum cross-version compatibility.
    """
    weight_data = []
    for w in weights:
        arr = np.array(w)
        weight_data.append(
            {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": arr.tobytes(),
            }
        )
    return pickle.dumps(weight_data, protocol=2)


def deserialize_weights(data):
    """Deserialize weights from raw-bytes format back to NumPy arrays."""
    weight_data = pickle.loads(data)
    weights = []
    for item in weight_data:
        arr = np.frombuffer(item["data"], dtype=np.dtype(item["dtype"]))
        arr = arr.reshape(item["shape"])
        # Return a writable copy (frombuffer returns read-only)
        weights.append(arr.copy())
    return weights


if __name__ == "__main__":
    # Test model building
    model = build_qcnn_model()
    model.summary()

    # Test weight serialization (using flat weights, same as federated protocol)
    weights = get_flat_weights(model)
    serialized = serialize_weights(weights)
    print(f"Serialized weight size: {len(serialized)} bytes")

    # Test deserialization
    weights_restored = deserialize_weights(serialized)
    print(f"Restored {len(weights_restored)} weight arrays")

    # Verify round-trip correctness
    for i, (orig, restored) in enumerate(zip(weights, weights_restored)):
        assert np.array_equal(orig, restored), f"Mismatch at weight {i}"
    print("Round-trip verification passed!")
