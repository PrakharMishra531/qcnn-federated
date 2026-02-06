import socket
import threading
import time
import numpy as np
import logging
import argparse
import pickle
import os
from datetime import datetime
from model import (
    build_qcnn_model,
    get_flat_weights,
    set_weights_from_flat,
    serialize_weights,
    deserialize_weights,
)
from config import (
    SERVER_HOST,
    SERVER_PORT,
    BUFFER_SIZE,
    NUM_CLIENTS,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    FEDERATED_DATA_PATH,
)

# Create output directory for logs and models
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "logs_and_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "server_federated_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FederatedServer:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT, num_clients=NUM_CLIENTS):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.server_socket = None
        self.clients = []
        self.client_info = {}
        self.global_model = None
        self.global_weights = None
        self.round = 0
        self.client_updates = {}
        self.lock = threading.Lock()

    def start(self):
        """Start the federated learning server."""
        # Create and initialize global model
        logger.info("Initializing global QCNN model...")
        self.global_model = build_qcnn_model()
        self.global_weights = get_flat_weights(self.global_model)

        # Start server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_clients)
        self.server_socket.settimeout(30)  # 30 second timeout for accepts

        logger.info(f"Server started on {self.host}:{self.port}")
        logger.info(f"Waiting for {self.num_clients} clients to connect...")

        # Wait for client connections
        self.wait_for_clients()

        # Start federated training
        self.run_federated_training()

    def wait_for_clients(self):
        """Wait for all clients to connect."""
        connected_clients = 0
        connection_attempts = 0
        max_attempts = 10  # Try for 5 minutes (30s * 10)

        while (
            connected_clients < self.num_clients and connection_attempts < max_attempts
        ):
            try:
                client_socket, address = self.server_socket.accept()
                connection_attempts += 1

                with self.lock:
                    client_id = connected_clients + 1
                    self.clients.append(client_socket)
                    self.client_info[client_id] = {
                        "address": address,
                        "connected": True,
                        "n_samples": 0,
                        "steps": 0,
                    }

                    # Send client ID
                    client_socket.send(str(client_id).encode())

                    logger.info(f"Client {client_id} connected from {address}")
                    connected_clients += 1

            except socket.timeout:
                logger.info(
                    f"Waiting for clients... ({connected_clients}/{self.num_clients})"
                )
                continue

        if connected_clients < self.num_clients:
            logger.warning(
                f"Only {connected_clients}/{self.num_clients} clients connected"
            )
        else:
            logger.info(f"All {self.num_clients} clients connected successfully")

    def broadcast_global_weights(self):
        """Send current global model weights to all clients."""
        logger.info("Broadcasting global model weights to clients...")
        serialized_weights = serialize_weights(self.global_weights)
        total_sent = 0

        for i, client_socket in enumerate(self.clients):
            try:
                client_id = i + 1
                # Send weight size first
                size = len(serialized_weights)
                client_socket.send(str(size).encode().ljust(16))

                # Send weights in chunks if necessary
                sent = 0
                while sent < size:
                    chunk = serialized_weights[sent : sent + BUFFER_SIZE]
                    client_socket.send(chunk)
                    sent += len(chunk)

                total_sent += size
                logger.debug(f"Sent {size} bytes to client {client_id}")

            except Exception as e:
                logger.error(f"Error sending weights to client {client_id}: {e}")
                self.client_info[client_id]["connected"] = False

        logger.info(
            f"Broadcast complete: {total_sent} bytes sent to {len(self.clients)} clients"
        )

    def receive_client_updates(self):
        """Receive updates from all clients."""
        logger.info("Waiting for client updates...")
        received_updates = {}
        start_time = time.time()

        for i, client_socket in enumerate(self.clients):
            client_id = i + 1

            try:
                # Receive metadata (n_samples, steps)
                metadata = client_socket.recv(64).decode().strip()
                n_samples, steps = map(int, metadata.split(","))

                # Receive weight update size
                size_data = client_socket.recv(16).decode().strip()
                if not size_data:
                    logger.error(f"Client {client_id} sent empty size data")
                    continue

                size = int(size_data)

                # Receive weight update
                data = b""
                while len(data) < size:
                    chunk = client_socket.recv(min(BUFFER_SIZE, size - len(data)))
                    if not chunk:
                        break
                    data += chunk

                # Deserialize weight delta
                weight_delta = deserialize_weights(data)

                received_updates[client_id] = {
                    "weight_delta": weight_delta,
                    "n_samples": n_samples,
                    "steps": steps,
                    "client_socket": client_socket,
                }

                self.client_info[client_id]["n_samples"] = n_samples
                self.client_info[client_id]["steps"] = steps
                logger.info(
                    f"Received update from client {client_id}: "
                    f"{n_samples} samples, {steps} steps"
                )

            except Exception as e:
                logger.error(f"Error receiving update from client {client_id}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Received updates from {len(received_updates)} clients in {elapsed:.2f}s"
        )

        return received_updates

    def fednova_aggregation(self, client_updates):
        """
        Perform FedNova aggregation.

        FedNova normalizes updates based on local training steps to handle
        client heterogeneity in computation.
        """
        logger.info("Performing FedNova aggregation...")

        total_samples = sum(update["n_samples"] for update in client_updates.values())
        client_weights = {
            k: v["n_samples"] / total_samples for k, v in client_updates.items()
        }

        # Compute mean number of steps
        mean_steps = sum(
            client_weights[k] * client_updates[k]["steps"]
            for k in client_updates.keys()
        )

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Mean steps: {mean_steps:.2f}")

        # Normalize and aggregate updates
        aggregated_update = None
        for client_id, update in client_updates.items():
            weight_delta = update["weight_delta"]
            n_samples = update["n_samples"]
            steps = update["steps"]

            # Normalize: (delta / steps) * (n_samples / total_samples)
            normalized_delta = [
                (delta / steps) * (n_samples / total_samples) for delta in weight_delta
            ]

            if aggregated_update is None:
                aggregated_update = normalized_delta
            else:
                aggregated_update = [
                    agg + norm for agg, norm in zip(aggregated_update, normalized_delta)
                ]

        # Scale by mean_steps
        final_update = [update * mean_steps for update in aggregated_update]

        # Update global weights
        self.global_weights = [
            gw + fu for gw, fu in zip(self.global_weights, final_update)
        ]

        logger.info("FedNova aggregation complete")

        # Apply updated weights to global model
        self.global_model = set_weights_from_flat(
            self.global_model, self.global_weights
        )

        return self.global_weights

    def run_federated_training(self):
        """Run the main federated training loop."""
        logger.info("=" * 60)
        logger.info("Starting Federated Training")
        logger.info(f"Number of rounds: {NUM_ROUNDS}")
        logger.info(f"Local epochs per round: {LOCAL_EPOCHS}")
        logger.info(f"Number of clients: {self.num_clients}")
        logger.info("=" * 60)

        for round_num in range(1, NUM_ROUNDS + 1):
            self.round = round_num
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Round {round_num}/{NUM_ROUNDS}")
            logger.info(f"{'=' * 60}")

            # Step 1: Broadcast global weights
            self.broadcast_global_weights()

            # Step 2: Receive client updates
            client_updates = self.receive_client_updates()

            if len(client_updates) == 0:
                logger.warning("No updates received from clients. Skipping round.")
                continue

            # Step 3: Aggregate using FedNova
            self.fednova_aggregation(client_updates)

            # Step 4: Log progress
            self.log_progress()

        # Save final model
        self.save_final_model()

        # Cleanup
        self.shutdown()

    def log_progress(self):
        """Log training progress."""
        logger.info(f"Round {self.round} completed")

        # Log client statistics
        for client_id, info in self.client_info.items():
            if info["connected"]:
                logger.info(
                    f"  Client {client_id}: {info['n_samples']} samples, "
                    f"{info['steps']} steps"
                )

    def save_final_model(self):
        """Save the final global model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(OUTPUT_DIR, f"global_model_fednova_{timestamp}.h5")

        self.global_model.save(model_path)
        logger.info(f"Final global model saved to: {model_path}")

        # Also save weights separately using pickle (list of arrays)
        weights_path = os.path.join(
            OUTPUT_DIR, f"global_weights_fednova_{timestamp}.pkl"
        )
        with open(weights_path, "wb") as f:
            pickle.dump(self.global_weights, f)
        logger.info(f"Global weights saved to: {weights_path}")

    def shutdown(self):
        """Shutdown the server and close connections."""
        logger.info("Shutting down server...")

        for client_socket in self.clients:
            try:
                client_socket.close()
            except:
                pass

        if self.server_socket:
            self.server_socket.close()

        logger.info("Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument(
        "--host", type=str, default=SERVER_HOST, help="Server host address"
    )
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument(
        "--clients", type=int, default=NUM_CLIENTS, help="Number of clients"
    )

    args = parser.parse_args()

    server = FederatedServer(host=args.host, port=args.port, num_clients=args.clients)
    server.start()


if __name__ == "__main__":
    main()
