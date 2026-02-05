# Understanding the Alzheimer's Detection Project Using Quanvolutional Neural Networks with Federated Training

## Overview of the Project 

This project focuses on Alzheimer's disease detection using **quanvolutional neural networks (QCNNs)** in a **federated learning (FL)** setting. Quanvolutional networks combine classical convolutional neural networks (CNNs) with quantum computing by applying quantum circuits to image patches to extract quantum features, which are then fed into classical neural networks for classification. Federated learning enables training across decentralized clients (e.g., hospitals) without sharing raw data, preserving privacy.

Key concepts include:
- **Quantum Feature Extraction**: Images are divided into patches, each processed by a quantum circuit (using PennyLane) to generate expectation values as features.
- **Federated Learning Algorithms**: 
  - **FedAvg**: Averages model updates from clients weighted by their data sizes.
  - **FedNova**: Normalizes updates based on local training steps to handle client heterogeneity.
- **Alzheimer's Dataset**: Classified into four categories (MildDemented, ModerateDemented, NonDemented, VeryMildDemented) from MRI images.
- **Evaluation**: Uses accuracy, loss, and classification reports (precision, recall, F1-score).

The notebooks implement and compare QCNNs vs. CNNs, and federated training setups with 3 or 5 clients. All use TensorFlow/Keras for models, PennyLane for quantum simulations, and scikit-learn for utilities.

---

## 1. Notebook: `Qcnn_vs_cnn.ipynb` (Root Directory)

### Purpose
This notebook compares a QCNN (using pre-computed quantum features) and a CNN (using raw images) on a centralized dataset for Alzheimer's detection. It trains both models on the same data split and evaluates their performance, highlighting quantum feature benefits.

### Imports
- **Core Libraries**: `os`, `numpy`, `tensorflow` (as `K` for Keras), `sklearn.model_selection` (for train-test split), `cv2` (OpenCV for image processing), `tqdm` (progress bars), `matplotlib.pyplot` (plotting).
- **Quantum**: `pennylane` (for quantum circuits and feature extraction).
- **Evaluation**: `sklearn.metrics` (classification report, confusion matrix), `seaborn` (heatmaps).

### Data Handling
- **Paths**: Loads quantum features (`quantum_features_combined.npy`, `labels_combined.npy`) from a Kaggle dataset (`/kaggle/input/qfeatures-and-mri/Quantum_features_/Combined`). Raw images from `/kaggle/input/qfeatures-and-mri/Quantum_features_/data_set/AugmentedAlzheimerDataset`.
- **Preprocessing**: 
  - Quantum features: Loaded directly (shape inferred from `X_q.shape[1]`, e.g., flattened patch expectations).
  - Raw images: Loaded via `load_dataset()` function, resized to 64x64, grayscale, normalized to [0,1], and converted to (64,64,1) for CNN.
- **Splitting**: Stratified train-test split (60% train, 40% test initially, then 80% train, 20% test for CNN). Classes: 4 (MildDemented, ModerateDemented, NonDemented, VeryMildDemented).
- **Dataset Size**: ~735 samples per class (total ~2940).

### Model Architectures
- **QCNN**: Multi-layer perceptron (MLP) with input shape matching quantum features (e.g., 4096 dimensions from 32x32 patches x 4 expectations). Layers: Dense(1024) → LeakyReLU(0.1) → BatchNorm → Dropout(0.3) → Dense(512) → ... → Dense(4, softmax). Uses He uniform initialization, Adam optimizer (LR=0.0005), sparse categorical cross-entropy.
- **CNN**: Convolutional network with input (64,64,1). Starts with Conv2D(1, (2,2), relu, bias=True), then Flatten → BatchNorm → Dense layers mirroring QCNN (1024 → 512 → 256 → 128 → 4). Same optimizer and loss.
- **Key Difference**: QCNN processes quantum-transformed features (no convolution on images); CNN applies light convolution directly on pixels.

### Training Procedures
- **QCNN**: Fit on quantum features (`Xq_train`, `yq_train`) for 15 epochs, batch size 32, validation on test set. Uses early stopping implicitly via validation.
- **CNN**: Fit on raw images (`Xc_train`, `yc_train`) for 15 epochs, same batch size.
- **Hyperparameters**: LR=0.0005, epochs=15, batch size=32.
- **Output**: Training history with accuracy/loss plots.

### Evaluation Metrics
- **Test Evaluation**: Loss, accuracy on held-out test set.
- **Classification Report**: Precision, recall, F1-score per class.
- **Confusion Matrix**: Heatmap (e.g., QCNN: 89.12% accuracy; CNN: 87.21%).
- **Plots**: Accuracy/loss curves for train/validation.

### Key Algorithms/Concepts
- **Quantum Circuit**: `quantum_patch_circuit()` uses 4 qubits, RX rotations on patches, CNOTs for entanglement, measures Pauli-Z expectations.
- **Feature Extraction**: `extract_q_features()` divides images into 2x2 patches, applies quantum circuit, flattens outputs.
- **ML Concepts**: Batch normalization, dropout for regularization; stratified sampling for class balance.

### Summary
QCNN outperforms CNN (89.1% vs. 87.2% accuracy), suggesting quantum features capture better patterns. Notebook demonstrates centralized training; quantum preprocessing reduces input dimensionality while preserving features.

---

## 2. Notebook: `5_client implimentation/Qcnn_vs_cnn.ipynb`

### Purpose
Similar to the first notebook but adapted for a 5-client federated setup. Trains QCNN and CNN models per client, evaluates on test data, and compares performance. Demonstrates QCNN/CNN comparison in a decentralized context without actual federated aggregation.

### Imports
- Identical to the first notebook: Core libraries, PennyLane, scikit-learn, matplotlib, seaborn.

### Data Handling
- **Paths**: Test data from `/kaggle/input/qcnn-vs-cnn/Quantum_features/test`.
- **Preprocessing**: 
  - Quantum features: Extracted on-the-fly for test images using `extract_q_features()` (similar to first notebook).
  - Raw images: Loaded, resized to 64x64, normalized.
- **Client Data**: Implies pre-split data (e.g., 5 clients), but notebook focuses on test evaluation. Loads test images (735 samples, balanced).
- **Features**: Quantum features shape (400, 4096); raw images (400, 64, 64, 1).

### Model Architectures
- **QCNN**: Same MLP as first notebook (Dense layers with LeakyReLU, dropout, batch norm). Loaded from saved model (`fedavg_qcnn_model.h5` or `fednova_qcnn_model.h5`).
- **CNN**: Same convolutional architecture. Loaded from saved model (`cnn_model_15epochs.h5`).
- **Concise Summary**: `concise_summary()` prints layer details and parameters.

### Training Procedures
- **No Training Here**: Models are pre-trained (assumed from federated runs). Notebook loads models and evaluates.
- **Test Extraction**: Quantum features extracted for 400 test images using progress bars.

### Evaluation Metrics
- **Test Results**: QCNN: 90.5% accuracy, 0.41 loss; CNN: 87.21% accuracy, 0.479 loss (from earlier run).
- **Classification Report**: QCNN shows better recall for NonDemented; CNN better for VeryMildDemented.
- **Confusion Matrix**: Seaborn heatmaps (QCNN: blues; CNN: oranges).
- **Plots**: Accuracy/loss curves from training history (assumed loaded).

### Key Algorithms/Concepts
- **Quantum Feature Extraction**: Identical to first notebook; processes test images.
- **Model Loading**: Uses Keras `load_model()` for pre-trained weights.
- **ML Concepts**: Evaluation on unseen test set; confusion matrix for error analysis.

### Summary
Builds on the first notebook by evaluating in a client-based context. QCNN again outperforms CNN, confirming quantum features' efficacy. Notebook serves as evaluation script for federated models.

---

## 3. Notebook: `5_client implimentation/FedAVG_and_FedNova.ipynb`

### Purpose
Implements and compares FedAvg and FedNova algorithms in a 5-client federated setup using QCNNs on quantum features. Trains a global model across rounds, evaluates per-round and per-client performance, and saves models.

### Imports
- **Core**: `os`, `numpy`, `tensorflow` (Keras), `sklearn.model_selection`, `matplotlib.pyplot`, `tqdm`.
- **Quantum**: `pennylane`.
- **Evaluation**: `sklearn.metrics`.

### Data Handling
- **Client Data**: 5 clients (`Client_1` to `Client_5`), each with quantum features (`quantum_features_realistic.npy`, `labels_realistic.npy`) and raw images (inferred).
- **Preprocessing**: Loads client data via `load_clients()`; splits 80% train, 20% test per client (stratified).
- **Dataset Size**: Varies per client (e.g., Client_1: ~11624; Client_2: ~7156); total ~23k samples.
- **Features**: Quantum features per client (shape varies, e.g., 4096 dimensions).

### Model Architectures
- **QCNN Global Model**: MLP with Dense(1024) → LeakyReLU → BatchNorm → Dropout(0.3) → Dense(512) → ... → Dense(4, softmax). Adam optimizer (LR=0.001, clipnorm=1.0), mixed precision (float16).
- **Key**: Uses `make_model()` for identical architectures per client and global.

### Training Procedures
- **FedAvg**: 
  - Clients train locally (20 epochs) on their data.
  - Aggregates updates: `w_new = sum(w_local * n_i / N)`.
  - 10 rounds; tracks global accuracy/loss.
- **FedNova**:
  - Accounts for local steps: Mean steps = weighted average of (local_epochs * steps_per_epoch).
  - Aggregates: `w_new = w_global + mean_steps * sum( (w_local - w_global) / steps_i * n_i / N )`.
  - Handles heterogeneity (e.g., client data sizes).
- **Federated Loop**: For each round, broadcast global weights, local training, aggregate, evaluate.
- **Hyperparameters**: LR=0.001, local epochs=20, rounds=10, batch=32.
- **Output**: Per-round plots (accuracy/loss); per-client accuracy over rounds.

### Evaluation Metrics
- **Global**: Accuracy/loss per round (FedAvg: up to 92.15%; FedNova: up to 92.07%).
- **Per-Client**: Validation accuracy post-local training.
- **Final Models**: Saved as HDF5 (e.g., `fedavg_qcnn_model.h5`).
- **Plots**: Global accuracy/loss curves; per-client lines.

### Key Algorithms/Concepts
- **FedAvg**: Simple averaging; assumes uniform local steps.
- **FedNova**: Normalized averaging for variable client participation/computation.
- **FL Concepts**: Decentralized training, privacy preservation, client heterogeneity.
- **Quantum**: Feature extraction cached per client.
- **ML Concepts**: Mixed precision for efficiency; stratified splits.

### Summary
Demonstrates FL algorithms with QCNNs. FedAvg and FedNova achieve similar high accuracy (~92%), with FedNova potentially more robust to heterogeneity. Notebook trains and evaluates federated models for 5 clients.

---

## 4. Notebook: `3_clients_impilementation/project_1/global_train.ipynb`

### Purpose
Implements federated training with 3 clients using FedNova (a variant of normalized averaging). Focuses on data distribution, quantum feature extraction, and training/evaluation. Includes data splitting script and test evaluation.

### Imports
- **Core**: `os`, `numpy`, `cv2`, `pennylane`, `tensorflow` (Keras), `sklearn.model_selection`, `matplotlib.pyplot`, `random`, `collections.Counter`, `tqdm`.

### Data Handling
- **Data Splitting**: Script distributes images from source (`data_set`) to 3 clients randomly but unevenly (for heterogeneity). Plots per-client class distributions.
- **Preprocessing**: 
  - Quantum features: Extracted per client using `extract_q_features()` (patches 2x2, 4 qubits, expectations).
  - Labels: Saved as `labels.npy`.
- **Client Data**: Client_1, Client_2, Client_3 with varying sizes (e.g., Client_1: ~11624; Client_2: ~7156; Client_3: ~8404).
- **Test Data**: Separate test set (400 samples), features extracted on-the-fly.

### Model Architectures
- **QCNN**: Larger MLP (Dense(1024) → LeakyReLU(0.1) → BatchNorm → Dropout(0.3) → Dense(512) → ... → Dense(4, softmax)). Adam (LR=0.001).
- **Key**: Identical across clients/global.

### Training Procedures
- **FedNova Implementation**: 
  - 5 rounds; clients train locally (20 epochs).
  - Aggregates deltas normalized by local steps: `aggregated_scaled = sum( (delta_i / s_i) * (n_i / N) )`; `w_new = w_global + mean_steps * aggregated_scaled`.
- **Loop**: Broadcast, local fit, aggregate, save global model.
- **Hyperparameters**: LR=0.001, local epochs=20, rounds=5, batch=32.
- **Output**: Prints round progress; saves global model (`global_model.h5`).

### Evaluation Metrics
- **Global Test**: Accuracy 91.92%, loss 0.252 (on 400 test samples).
- **Per-Round**: Training logs (accuracy/loss per epoch).
- **Plots**: Class distributions per client (bar charts).

### Key Algorithms/Concepts
- **FedNova**: Detailed implementation with weight arithmetic functions (`weights_add`, `weights_scalar_mul`).
- **Data Heterogeneity**: Random splits create uneven client distributions.
- **Quantum Extraction**: Cached per client; tqdm for progress.
- **ML Concepts**: Batch norm, dropout; evaluation on test set.

### Summary
Focuses on 3-client FL with FedNova, emphasizing data preparation and quantum feature handling. Achieves ~92% test accuracy, demonstrating effective federated QCNN training. Includes utility scripts for distribution and testing.