import os

# Dataset Configuration
DATASET_PATH = os.path.join("..", "dataset", "AugmentedAlzheimerDataset")
FEDERATED_DATA_PATH = os.path.join(
    "..", "quantum_features"
)  # Pre-extracted features from Jupyter notebook

# Model Configuration
IMG_SIZE = 64
PATCH_SIZE = 2
NUM_QUBITS = 4
NUM_CLASSES = 4

# QCNN Architecture
INPUT_DIM = (IMG_SIZE // PATCH_SIZE) ** 2 * NUM_QUBITS  # 1024 * 4 = 4096
DENSE_LAYERS = [1024, 512, 256, 128]
DROPOUT_RATES = [0.3, 0.25, 0.15, 0.0]
LEARNING_RATE = 0.0001  # reduced to avoid collapse

# Federated Configuration
# NUM_CLIENTS = 5
# NUM_ROUNDS = 10
# LOCAL_EPOCHS = 20
# BATCH_SIZE = 32

NUM_CLIENTS = 1
NUM_ROUNDS = 3
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

# Communication Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
BUFFER_SIZE = 65536  # 64KB recv buffer for socket reads
CHUNK_SIZE = 1048576  # 1MB chunks for ACK-based transfer protocol
SOCKET_TIMEOUT = 300  # 5-minute timeout for socket operations

# IID vs Non-IID Configuration
IID_MODE = False  # Set to True for IID split, False for Non-IID

# Logging Configuration
LOG_FILE = "federated_training.log"
