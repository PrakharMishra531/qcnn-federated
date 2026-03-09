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
    CHUNK_SIZE,
    SOCKET_TIMEOUT,
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


# ---------------------------------------------------------------------------
# Shared helpers: chunked send/receive with ACK flow-control
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
# Federated Server
# ---------------------------------------------------------------------------


class FederatedServer:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT, num_clients=NUM_CLIENTS):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.server_socket = None
        self.clients = []  # list of client sockets
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

                # Tune the accepted client socket immediately
                configure_socket(client_socket)

                with self.lock:
                    client_id = connected_clients + 1
                    self.clients.append(client_socket)
                    self.client_info[client_id] = {
                        "address": address,
                        "connected": True,
                        "n_samples": 0,
                        "steps": 0,
                        "ready": False,
                    }

                    # Send client ID
                    client_socket.sendall(str(client_id).encode())

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

        # Wait for all clients to be READY (data loaded, model initialized)
        logger.info("Waiting for clients to be READY...")
        self.wait_for_client_ready()

    def wait_for_client_ready(self):
        """Wait for all clients to send READY signal (after data load + model init)."""
        ready_count = 0

        for idx, client_socket in enumerate(self.clients):
            client_id = idx + 1
            try:
                # Wait for READY message (8 bytes, padded)
                ready_msg = _recv_exact(client_socket, 8)
                msg = ready_msg.decode().strip()
                if msg == "READY":
                    ready_count += 1
                    self.client_info[client_id]["ready"] = True
                    logger.info(f"Client {client_id} is READY")
                else:
                    logger.warning(f"Unexpected message from client {client_id}: {msg}")
            except Exception as e:
                logger.error(f"Error waiting for READY from client {client_id}: {e}")

        logger.info(f"{ready_count}/{len(self.clients)} clients are READY")

    # ------------------------------------------------------------------
    # Weight broadcast (server -> clients)
    # ------------------------------------------------------------------

    def broadcast_global_weights(self):
        """Broadcast the current global weights to all connected clients.

        Uses chunked transfer with per-chunk ACK for flow control.
        Each client is handled independently so one failure doesn't
        affect the others.
        """
        logger.info("Broadcasting global model weights to clients...")

        serialized_weights = serialize_weights(self.global_weights)
        size = len(serialized_weights)
        logger.info(f"Serialized weight payload: {size:,} bytes")

        success_count = 0
        failed_clients = []

        for idx, client_socket in enumerate(self.clients):
            client_id = idx + 1
            try:
                send_chunked(client_socket, serialized_weights)
                success_count += 1
                logger.info(f"  Sent weights to client {client_id}")
            except Exception as e:
                logger.error(f"  Error sending weights to client {client_id}: {e}")
                self.client_info[client_id]["connected"] = False
                failed_clients.append(client_id)

        logger.info(
            f"Broadcast complete: {size:,} bytes sent to "
            f"{success_count}/{len(self.clients)} clients."
        )

        # Remove failed clients so we don't wait for them later
        for cid in failed_clients:
            idx = cid - 1
            try:
                self.clients[idx].close()
            except Exception:
                pass

        return success_count > 0

    # ------------------------------------------------------------------
    # Receive client updates (clients -> server)
    # ------------------------------------------------------------------

    def receive_client_updates(self):
        """Receive updates from all connected clients.

        Protocol per client:
            1. 64-byte metadata: "n_samples,steps" (left-justified, padded)
            2. Chunked weight-delta payload (with ACK flow-control)
        """
        logger.info("Waiting for client updates...")
        received_updates = {}
        start_time = time.time()

        for idx, client_socket in enumerate(self.clients):
            client_id = idx + 1
            if not self.client_info[client_id]["connected"]:
                continue

            try:
                # 1. 64-byte metadata
                meta_raw = _recv_exact(client_socket, 64)
                metadata = meta_raw.decode().strip()
                n_samples, steps = map(int, metadata.split(","))

                # 2. Chunked weight-delta payload
                data = recv_chunked(client_socket)

                # 3. Deserialize
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
                    f"{n_samples} samples, {steps} steps, "
                    f"{len(data):,} bytes"
                )

            except Exception as e:
                logger.error(f"Error receiving update from client {client_id}: {e}")
                self.client_info[client_id]["connected"] = False

        elapsed = time.time() - start_time
        logger.info(
            f"Received updates from {len(received_updates)} client(s) in {elapsed:.1f}s"
        )
        return received_updates

    # ------------------------------------------------------------------
    # FedNova Aggregation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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
            if not self.broadcast_global_weights():
                logger.error("Broadcast failed to all clients. Aborting.")
                break

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
            except Exception:
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
