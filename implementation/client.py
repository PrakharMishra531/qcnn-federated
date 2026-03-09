import argparse
import logging
import os
import socket
import time

import numpy as np

# ---------------------------------------------------------------------------
# TF compat patch: older TF builds on Jetson Nano are missing this attribute.
# Must run before any model.fit() call.
# ---------------------------------------------------------------------------
try:
    import tensorflow.compat.v2 as _tf2

    _dist = _tf2.__internal__.distribute
    if not hasattr(_dist, "strategy_supports_no_merge_call"):
        _dist.strategy_supports_no_merge_call = lambda: False
except Exception:
    pass
from config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    CHUNK_SIZE,
    SOCKET_TIMEOUT,
    FEDERATED_DATA_PATH,
    LOCAL_EPOCHS,
    NUM_ROUNDS,
    SERVER_HOST,
    SERVER_PORT,
)
from model import (
    build_qcnn_model,
    compute_weight_delta,
    deserialize_weights,
    get_flat_weights,
    serialize_weights,
    set_weights_from_flat,
)

# Create output directory for logs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "logs_and_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "client_federated_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers: chunked send/receive with ACK flow-control
# (Must stay in sync with the identical helpers in server.py)
# ---------------------------------------------------------------------------


def _recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from sock. Raises ConnectionError on drop."""
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection dropped: received {len(buf)}/{num_bytes} bytes."
            )
        buf.extend(chunk)
    return bytes(buf)


def send_chunked(sock, payload, chunk_size=CHUNK_SIZE):
    """Send *payload* in chunks, waiting for a 3-byte ACK after each chunk.

    Protocol (sender side):
        1. Send 16-byte left-justified size header.
        2. For each chunk:
           a. sendall(chunk)
           b. recv 3-byte ACK from receiver.
    """
    size = len(payload)
    header = str(size).ljust(16).encode()
    sock.sendall(header)

    offset = 0
    while offset < size:
        end = min(offset + chunk_size, size)
        sock.sendall(payload[offset:end])
        # Wait for ACK before sending next chunk
        ack = _recv_exact(sock, 3)
        if ack != b"ACK":
            raise ConnectionError(f"Expected ACK, got {ack!r}")
        offset = end


def recv_chunked(sock, chunk_size=CHUNK_SIZE, buffer_size=BUFFER_SIZE):
    """Receive a chunked payload, sending a 3-byte ACK after each chunk.

    Protocol (receiver side):
        1. Receive 16-byte size header.
        2. While bytes remaining:
           a. recv up to min(chunk_size, remaining) bytes.
           b. sendall(b'ACK').
        3. Return complete payload bytes.
    """
    header = _recv_exact(sock, 16)
    size = int(header.decode().strip())

    data = bytearray()
    while len(data) < size:
        # How many bytes belong to the current chunk?
        chunk_remaining = min(chunk_size, size - len(data))
        # Receive the whole chunk (may arrive in several TCP segments)
        while chunk_remaining > 0:
            piece = sock.recv(min(buffer_size, chunk_remaining))
            if not piece:
                raise ConnectionError(
                    f"Connection dropped: received {len(data)}/{size} bytes."
                )
            data.extend(piece)
            chunk_remaining -= len(piece)
        # Acknowledge this chunk
        sock.sendall(b"ACK")

    return bytes(data)


def configure_socket(sock):
    """Apply TCP_NODELAY, timeout, and enlarged buffers to *sock*."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass  # Some OS/containers do not allow enlarging buffers


# ---------------------------------------------------------------------------
# Federated Client
# ---------------------------------------------------------------------------


class FederatedClient:
    def __init__(self, client_id, server_host=SERVER_HOST, server_port=SERVER_PORT):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.local_model = None
        self.global_weights = None
        self.local_weights = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.n_samples = 0
        self.steps_per_epoch = 0

    def connect_to_server(self):
        """Connect to the federated learning server."""
        logger.info(f"Connecting to server at {self.server_host}:{self.server_port}...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(self.socket)
        self.socket.connect((self.server_host, self.server_port))

        # Receive client ID from server
        client_id = int(self.socket.recv(1024).decode())
        assert client_id == self.client_id, (
            f"Expected client {self.client_id}, got {client_id}"
        )
        logger.info(f"Connected to server as Client {client_id}")

    def load_local_data(self):
        """Load pre-extracted quantum features and labels from .npy files."""
        client_path = os.path.join(FEDERATED_DATA_PATH, f"client_{self.client_id}")
        train_features_path = os.path.join(client_path, "train_features.npy")
        train_labels_path = os.path.join(client_path, "train_labels.npy")
        test_features_path = os.path.join(client_path, "test_features.npy")
        test_labels_path = os.path.join(client_path, "test_labels.npy")

        if not os.path.exists(train_features_path):
            logger.error(f"Train features not found at {train_features_path}")
            logger.error("Please run quantum_feature_extraction.ipynb first")
            raise FileNotFoundError(f"Train features not found: {train_features_path}")

        logger.info(f"Loading pre-extracted features from {train_features_path}")
        self.X_train = np.load(train_features_path)
        self.y_train = np.load(train_labels_path)
        self.X_test = np.load(test_features_path)
        self.y_test = np.load(test_labels_path)

        self.n_samples = len(self.X_train)
        self.steps_per_epoch = (self.n_samples + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Loaded: {self.n_samples} train, {len(self.X_test)} test samples")

    def initialize_model(self):
        """Initialize local QCNN model."""
        logger.info("Initializing local QCNN model...")
        self.local_model = build_qcnn_model()
        self.local_weights = get_flat_weights(self.local_model)

    def send_ready(self):
        """Send READY signal to server after data load + model init."""
        try:
            self.socket.sendall(b"READY   ")  # 8 bytes, padded with spaces
            logger.info("Sent READY signal to server")
        except Exception as e:
            logger.error(f"Error sending READY: {e}")

    # ------------------------------------------------------------------
    # Receive global weights (server -> client)  [chunked ACK protocol]
    # ------------------------------------------------------------------

    def receive_global_weights(self):
        """Receive global model weights from server using chunked ACK protocol."""
        logger.info("Waiting for global weights from server...")
        try:
            data = recv_chunked(self.socket)

            self.global_weights = deserialize_weights(data)
            self.local_model = set_weights_from_flat(
                self.local_model, self.global_weights
            )
            self.local_weights = self.global_weights

            logger.info(f"Received global weights: {len(data):,} bytes")
            return True
        except Exception as e:
            logger.error(f"Error receiving global weights: {e}")
            return False

    # ------------------------------------------------------------------
    # Local training  (no sklearn -- Nano-compatible)
    # ------------------------------------------------------------------

    def train_locally(self):
        """Perform local training on client data."""
        logger.info(f"Starting local training for {LOCAL_EPOCHS} epochs...")

        # Ensure data is 2D (samples, features)
        X_train_processed = self.X_train
        if len(X_train_processed.shape) > 2:
            X_train_processed = X_train_processed.reshape(
                X_train_processed.shape[0], -1
            )

        # Flatten test data the same way
        X_test_processed = self.X_test
        if len(X_test_processed.shape) > 2:
            X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], -1)

        # Manual StandardScaler (no sklearn dependency on Nano)
        mean = np.mean(X_train_processed, axis=0)
        std = np.std(X_train_processed, axis=0)
        std[std == 0] = 1e-7
        X_train_processed = (X_train_processed - mean) / std
        X_test_processed = (X_test_processed - mean) / std  # same transform

        # Manual class weight computation (no sklearn dependency on Nano)
        unique_classes = np.unique(self.y_train)
        n_samples = len(self.y_train)
        n_classes = len(unique_classes)
        class_counts = np.bincount(self.y_train.astype(int))
        class_weight_dict = {}
        for cls in unique_classes:
            cls_int = int(cls)
            class_weight_dict[cls_int] = n_samples / (n_classes * class_counts[cls_int])

        logger.info(f"Class Weights applied: {class_weight_dict}")

        history = self.local_model.fit(
            X_train_processed,
            self.y_train,
            validation_data=(X_test_processed, self.y_test),
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            verbose=1,
        )

        self.local_weights = get_flat_weights(self.local_model)

        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        logger.info(f"Local training completed:")
        logger.info(f"  Train accuracy: {final_train_acc:.4f}")
        logger.info(f"  Validation accuracy: {final_val_acc:.4f}")
        logger.info(f"  Train loss: {final_train_loss:.4f}")
        logger.info(f"  Validation loss: {final_val_loss:.4f}")
        return history

    # ------------------------------------------------------------------
    # Compute & send update (client -> server)  [chunked ACK protocol]
    # ------------------------------------------------------------------

    def compute_update(self):
        """Compute weight update (delta = local - global)."""
        logger.info("Computing weight update...")
        weight_delta = compute_weight_delta(self.local_weights, self.global_weights)
        local_steps = LOCAL_EPOCHS * self.steps_per_epoch

        logger.info(
            f"Update computed: {len(weight_delta)} layers, "
            f"{local_steps} local steps, {self.n_samples} samples"
        )
        return weight_delta, local_steps

    def send_update(self, weight_delta, local_steps):
        """Send weight update to server using chunked ACK protocol.

        Protocol:
            1. 64-byte metadata: "n_samples,steps" (left-justified, padded)
            2. Chunked weight-delta payload (with ACK flow-control)
        """
        logger.info("Sending update to server...")
        try:
            # 1. 64-byte metadata
            metadata = f"{self.n_samples},{local_steps}"
            self.socket.sendall(metadata.ljust(64).encode())

            # 2. Chunked payload
            serialized_delta = serialize_weights(weight_delta)
            send_chunked(self.socket, serialized_delta)

            logger.info(f"Update sent: {len(serialized_delta):,} bytes")
            return True
        except Exception as e:
            logger.error(f"Error sending update: {e}")
            return False

    # ------------------------------------------------------------------
    # Main federated loop
    # ------------------------------------------------------------------

    def run_federated_training(self):
        """Run the main federated learning loop on client side."""
        logger.info("=" * 60)
        logger.info(f"Client {self.client_id} Starting Federated Training")
        logger.info("=" * 60)

        self.connect_to_server()
        self.load_local_data()
        self.initialize_model()
        self.send_ready()

        for round_num in range(1, NUM_ROUNDS + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Round {round_num}/{NUM_ROUNDS}")
            logger.info(f"{'=' * 60}")

            # Receive global weights
            success = self.receive_global_weights()
            if not success:
                logger.error("Failed to receive global weights. Aborting.")
                break

            # Train locally
            self.train_locally()

            # Compute and send update
            weight_delta, local_steps = self.compute_update()
            success = self.send_update(weight_delta, local_steps)
            if not success:
                logger.error("Failed to send update to server. Aborting.")
                break

        logger.info("=" * 60)
        logger.info("Federated training completed!")
        logger.info("=" * 60)
        self.shutdown()

    def shutdown(self):
        """Close socket connection."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("Socket connection closed")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--id", type=int, required=True, help="Client ID (1-5)")
    parser.add_argument(
        "--host", type=str, default=SERVER_HOST, help="Server host address"
    )
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")

    args = parser.parse_args()

    if args.id < 1 or args.id > 5:
        logger.error("Client ID must be between 1 and 5")
        return

    client = FederatedClient(
        client_id=args.id, server_host=args.host, server_port=args.port
    )
    client.run_federated_training()


if __name__ == "__main__":
    main()
