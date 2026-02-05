# Federated QCNN Training for Alzheimer's Detection

## Overview

This implementation provides a modular, production-ready federated learning system for Alzheimer's disease detection using Quanvolutional Neural Networks (QCNNs) with FedNova aggregation.

**NEW WORKFLOW**: Quantum features are pre-extracted via Jupyter notebook, then federated training runs on Jetson Nano devices using only the pre-computed `.npy` files.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Pre-computation (Powerful Machine)                  │
│                                                                 │
│   Jupyter Notebook: quantum_feature_extraction.ipynb            │
│   ↓                                                             │
│   Extract features for all clients                              │
│   ↓                                                             │
│   quantum_features/client_X/                                     │
│   ├── features.npy  (4096-dim per image)                        │
│   ├── labels.npy                                               │
│   └── config.json                                               │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼ Copy to Jetson Nano
┌─────────────────────────────────────────────────────────────────┐
│              Federated Learning (Jetson Nano)                    │
│                                                                 │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│    │   Client 1   │     │   Client 2   │     │   Client 3   │ │
│    │  (Pre-computed│     │  (Pre-computed│     │  (Pre-computed│ │
│    │   features)   │     │   features)   │     │   features)   │ │
│    │      ↓       │     │      ↓       │     │      ↓       │ │
│    │   QCNN       │     │   QCNN       │     │   QCNN       │ │
│    │   Training   │     │   Training   │     │   Training   │ │
│    │      ↓       │     │      ↓       │     │      ↓       │ │
│    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘ │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│                         ┌──────┴──────┐                         │
│                         │   SERVER    │                         │
│                         │ (FedNova    │                         │
│                         │ Aggregation)│                         │
│                         └──────┴──────┘                         │
│                                │                                │
│                   ┌────────────┼────────────┐                 │
│                   ↓            ↓            ↓                   │
│            ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│            │Round 1   │  │Round 2   │  │Round N   │          │
│            │Weights   │  │Weights   │  │Weights   │          │
│            └──────────┘  └──────────┘  └──────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Pre-computed Quantum Features**: No quantum computation on Jetson Nano
- **Modular Design**: Each component is independent and reusable
- **Privacy-Preserving**: No raw data or quantum features shared between clients
- **FedNova Aggregation**: Handles client heterogeneity in computation
- **Jetson Nano Optimized**: Only loads pre-extracted `.npy` files

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- PennyLane 0.30+ (only for feature extraction on powerful machine)
- NumPy, Scikit-learn
- OpenCV for image processing

## Installation

```bash
cd implementation
pip install -r requirements.txt
```

## Project Structure

```
implementation/
├── config.py              # Configuration file
├── requirements.txt       # Dependencies
├── model.py              # QCNN model architecture
├── quantum_circuit.py    # PennyLane quantum circuit (for extraction only)
├── server.py             # Federated learning server
├── client.py             # Federated learning client
├── utils.py              # Utility functions
└── README.md             # This file

root_directory/
├── quantum_feature_extraction.ipynb  # Pre-computation notebook
└── quantum_features/                  # Pre-extracted features
    ├── client_1/
    │   ├── features.npy
    │   ├── labels.npy
    │   └── config.json
    ├── client_2/
    ├── client_3/
    ├── client_4/
    └── client_5/
```

## Usage

### Step 1: Pre-compute Quantum Features (Powerful Machine)

```bash
jupyter notebook quantum_feature_extraction.ipynb
```

1. Set `SAMPLES_PER_CLIENT = 2000` (or your desired number)
2. Run all cells
3. Features saved to `quantum_features/`

### Step 2: Deploy to Jetson Nano

Copy the `quantum_features/` folder to each Jetson Nano:

```bash
rsync -avh quantum_features/ user@jetson.local:/home/user/quantum_features/
```

### Step 3: Start Federated Training

**Start the server first:**

```bash
cd implementation
python server.py --host localhost --port 8080
```

**Then start each client (in separate terminals):**

```bash
cd implementation
python client.py --id 1 --host localhost --port 8080
python client.py --id 2 --host localhost --port 8080
python client.py --id 3 --host localhost --port 8080
python client.py --id 4 --host localhost --port 8080
python client.py --id 5 --host localhost --port 8080
```

## Configuration

Modify `config.py` to adjust:

- **Model Architecture**: Layer sizes, dropout rates
- **Training Parameters**: Learning rate, epochs, batch size
- **Federated Settings**: Number of rounds, clients
- **Communication**: Server host, port, buffer sizes

## Quantum Circuit

**NOTE**: Quantum circuit is only used during pre-computation on a powerful machine. Jetson Nano only loads the pre-extracted features.

```python
@qml.qnode(dev)
def quantum_patch_circuit(phi):
    # RX rotations on each qubit
    qml.RX(phi[0], wires=0)
    qml.RX(phi[1], wires=1)
    qml.RX(phi[2], wires=2)
    qml.RX(phi[3], wires=3)
    
    # CNOT entanglement layers
    qml.CNOT(wires=[0, 1])
    qml.RZ(np.pi / 2, wires=1)
    qml.CNOT(wires=[0, 1])
    
    qml.CNOT(wires=[2, 3])
    qml.RZ(np.pi / 2, wires=3)
    qml.CNOT(wires=[2, 3])
    
    qml.CNOT(wires=[1, 2])
    qml.RZ(np.pi / 2, wires=2)
    qml.CNOT(wires=[1, 2])
    
    # Return Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
```

## QCNN Architecture

```
Input Layer: 4096 features (pre-extracted quantum features)
    ↓
Dense(1024) → LeakyReLU → BatchNorm → Dropout(0.3)
    ↓
Dense(512) → LeakyReLU → BatchNorm → Dropout(0.25)
    ↓
Dense(256) → LeakyReLU → BatchNorm → Dropout(0.15)
    ↓
Dense(128) → LeakyReLU → BatchNorm
    ↓
Dense(4, softmax)  # 4 Alzheimer classes
```

## FedNova Aggregation

FedNova normalizes updates based on local training steps:

```
normalized_update_k = (delta_k / steps_k) * (n_k / total_samples)
mean_steps = Σ (n_k / total_samples) * steps_k
global_update = mean_steps * Σ normalized_update_k
new_global = old_global + global_update
```

This handles client heterogeneity where clients may perform different amounts of local computation.

## Logging

Training logs are saved to:
- `federated_training.log` (server)
- `client_federated_training.log` (clients)

Logs include:
- Round-by-round accuracy and loss
- Client participation statistics
- Communication size (bytes sent/received)

## Model Output

Final models are saved as:
- `global_model_fednova_{timestamp}.h5` (full Keras model)
- `global_weights_fednova_{timestamp}.npy` (weights only)

## Jetson Nano Storage Requirements

| Images/Client | Feature Size | Memory Usage |
|---------------|--------------|--------------|
| 500           | ~7.8 MB      | ~15 MB       |
| 1000          | ~15.6 MB     | ~30 MB       |
| 2000          | ~31.2 MB     | ~60 MB       |


---

My Instructions

once features are extracted:
    python server.py
    # Terminal 2-6 - Start 5 clients
    python client.py --id 1
    python client.py --id 2
    python client.py --id 3
    python client.py --id 4
    python client.py --id 5 

Evaluation - NOT Automatic. You need to run it manually.
What Happens After Training
When server.py completes all rounds, it automatically saves the model:
Saved files:
├── global_model_fednova_20260205_120000.h5    ← Trained model (Keras format)
└── global_weights_fednova_20260205_120000.pkl ← Weights only


After training completes, run:
cd implementation
python evaluate_model.py \
    --model global_model_fednova_*.h5 \
    --features ../quantum_features/test_features.npy \
    --labels ../quantum_features/test_labels.npy
Or let it auto-detect the latest model:
python evaluate_model.py \
    --features ../quantum_features/test_features.npy \
    --labels ../quantum_features/test_labels.npy


    The evaluate_model.py script produces:
| Output | Description |
|--------|-------------|
| confusion_matrix_*.png | Heatmap of predictions vs true labels |
| per_class_metrics_*.png | Bar chart of Precision, Recall, F1 per class |
| accuracy_by_class_*.png | Per-class accuracy |
| probability_distribution_*.png | Confidence distribution |
| evaluation_summary_*.png | Text summary of results |
| evaluation_report.txt | Detailed text report |



