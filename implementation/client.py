import socket
import time
import numpy as np
import logging
import argparse
import os
from model import (
    build_qcnn_model,
    get_flat_weights,
    set_weights_from_flat,
    serialize_weights,
    deserialize_weights,
    compute_weight_delta,
)
from config import (
    SERVER_HOST,
    SERVER_PORT,
    BUFFER_SIZE,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    FEDERATED_DATA_PATH,
    LOG_FILE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"client_{LOG_FILE}"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        self.X_val = None
        self.y_val = None
        self.n_samples = 0
        self.steps_per_epoch = 0

    def connect_to_server(self):
        """Connect to the federated learning server."""
        logger.info(f"Connecting to server at {self.server_host}:{self.server_port}...")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        
        # No train_test_split needed - already split!
        self.n_samples = len(self.X_train)
        self.steps_per_epoch = (self.n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
        logger.info(f"Loaded: {self.n_samples} train, {len(self.X_test)} test samples")

    def initialize_model(self):
        """Initialize local QCNN model."""
        logger.info("Initializing local QCNN model...")
        self.local_model = build_qcnn_model()
        self.local_weights = get_flat_weights(self.local_model)

    def receive_global_weights(self):
        """Receive global model weights from server."""
        logger.info("Waiting for global weights from server...")

        try:
            # Receive weight size
            size_data = self.socket.recv(16).decode().strip()
            if not size_data:
                logger.error("Received empty size data from server")
                return False

            size = int(size_data)

            # Receive weights
            data = b""
            while len(data) < size:
                chunk = self.socket.recv(min(BUFFER_SIZE, size - len(data)))
                if not chunk:
                    break
                data += chunk

            self.global_weights = deserialize_weights(data)

            # Update local model with global weights
            self.local_model = set_weights_from_flat(
                self.local_model, self.global_weights
            )
            self.local_weights = self.global_weights

            logger.info(f"Received global weights: {size} bytes")
            return True

        except Exception as e:
            logger.error(f"Error receiving global weights: {e}")
            return False

    def train_locally(self):
        """Perform local training on client data."""
        logger.info(f"Starting local training for {LOCAL_EPOCHS} epochs...")

        history = self.local_model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        # Get final local weights
        self.local_weights = get_flat_weights(self.local_model)

        # Log training metrics
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

    def compute_update(self):
        """Compute weight update (delta = local - global)."""
        logger.info("Computing weight update...")

        weight_delta = compute_weight_delta(self.local_weights, self.global_weights)

        # Calculate number of local steps
        local_steps = LOCAL_EPOCHS * self.steps_per_epoch

        logger.info(
            f"Update computed: {len(weight_delta)} layers, "
            f"{local_steps} local steps, {self.n_samples} samples"
        )

        return weight_delta, local_steps

    def send_update(self, weight_delta, local_steps):
        """Send weight update to server."""
        logger.info("Sending update to server...")

        try:
            # Send metadata: n_samples, local_steps
            metadata = f"{self.n_samples},{local_steps}"
            self.socket.send(metadata.encode().ljust(64))

            # Serialize and send weight delta
            serialized_delta = serialize_weights(weight_delta)
            size = len(serialized_delta)

            # Send size first
            self.socket.send(str(size).encode().ljust(16))

            # Send data in chunks
            sent = 0
            while sent < size:
                chunk = serialized_delta[sent : sent + BUFFER_SIZE]
                self.socket.send(chunk)
                sent += len(chunk)

            logger.info(f"Update sent: {size} bytes")
            return True

        except Exception as e:
            logger.error(f"Error sending update: {e}")
            return False

    def run_federated_training(self):
        """Run the main federated learning loop on client side."""
        logger.info("=" * 60)
        logger.info(f"Client {self.client_id} Starting Federated Training")
        logger.info("=" * 60)

        # Connect to server
        self.connect_to_server()

        # Load local data
        self.load_local_data()

        # Initialize model
        self.initialize_model()

        # Federated training loop
        for round_num in range(1, NUM_ROUNDS + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Round {round_num}/{NUM_ROUNDS}")
            logger.info(f"{'=' * 60}")

            # Step 1: Receive global weights
            success = self.receive_global_weights()
            if not success:
                logger.error("Failed to receive global weights. Skipping round.")
                continue

            # Step 2: Train locally
            self.train_locally()

            # Step 3: Compute update
            weight_delta, local_steps = self.compute_update()

            # Step 4: Send update to server
            success = self.send_update(weight_delta, local_steps)
            if not success:
                logger.error("Failed to send update to server.")
                continue

        # Training complete
        logger.info("=" * 60)
        logger.info("Federated training completed!")
        logger.info("=" * 60)

        # Cleanup
        self.shutdown()

    def shutdown(self):
        """Close socket connection."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("Socket connection closed")
            except:
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
