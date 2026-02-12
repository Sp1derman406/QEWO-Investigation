from typing import override
#WORKS 2 hidden layer wine
# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import time
from random import *

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from math import pi, floor, sqrt


# =========================
# CONFIG
# =========================
NUM_EPOCHS = 10

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64

RES_W1_SF = 3.5
RES_W2_SF = 3.5
RES_W3_SF = 3.5
overr = 3

SHOTS = 256
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-3

INIT_SEARCH_STD = 0.5
MIN_SEARCH_STD = 0.1
MAX_SEARCH_STD = 2.0


# =========================
# Shared initial weights
# =========================
np.random.seed(randint(20,60))

W1_INIT = np.random.uniform(-1, 1, (13, HIDDEN_SIZE_1))
W2_INIT = np.random.uniform(-1, 1, (HIDDEN_SIZE_1, HIDDEN_SIZE_2))
W3_INIT = np.random.uniform(-1, 1, (HIDDEN_SIZE_2, 3))


# =========================
# Grover primitives
# =========================
def oracle(num_qubits, marked_states):
    qc = QuantumCircuit(num_qubits + 1)
    for state in marked_states:
        binary = format(state, f'0{num_qubits}b')
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)
        qc.mcx(list(range(num_qubits)), num_qubits)
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)
    return qc.to_gate(label="Oracle")


def diffusion_operator(num_qubits):
    qc = QuantumCircuit(num_qubits + 1)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc.to_gate(label="Diffusion")


def quantum_min(num_qubits, marked_states, N, shots):
    M = len(marked_states)
    if M == 0:
        raise ValueError("No marked states.")
    k = max(1, floor((pi / 4) * sqrt(N / M)))

    qc = QuantumCircuit(num_qubits + 1, num_qubits)
    qc.h(range(num_qubits))
    qc.x(num_qubits)
    qc.h(num_qubits)

    og = oracle(num_qubits, marked_states)
    dg = diffusion_operator(num_qubits)

    for _ in range(k):
        qc.append(og, range(num_qubits + 1))
        qc.append(dg, range(num_qubits + 1))

    qc.measure(range(num_qubits), range(num_qubits))
    qc = qc.decompose(reps=10)

    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    best = max(counts, key=counts.get)
    return int(best, 2) % N


# =========================
# MLP (2 hidden layers)
# =========================
class MLP:
    def __init__(self):
        self.W1 = W1_INIT.copy()
        self.W2 = W2_INIT.copy()
        self.W3 = W3_INIT.copy()

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        h1 = np.tanh(x @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        out = self.softmax(h2 @ self.W3)
        return h1, h2, out


# =========================
# Grover layer optimizer
# =========================
def optimize_layer(
    mlp,
    layer_id,
    layer_input,
    y_true,
    weights,
    resolution_sf,
    search_std,
    shots
):
    W = weights.copy()
    std = max(np.std(W), 1e-6)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            low = W[i, j] - search_std * std
            high = W[i, j] + search_std * std
            if j == 0:
                resolution = 6
            else:
                resolution = int(-1*resolution_sf*(ce+reg)+10.35)

            resolution = max(min(resolution,10),3)

            resolution = overr

            print(f"using resolution{resolution}")
            candidates = np.linspace(low, high, resolution)

            losses = []
            for c in candidates:
                W_try = W.copy()
                W_try[i, j] = c

                if layer_id == 1:
                    h1 = np.tanh(layer_input @ W_try)
                    h2 = np.tanh(h1 @ mlp.W2)
                    out = mlp.softmax(h2 @ mlp.W3)
                elif layer_id == 2:
                    h2 = np.tanh(layer_input @ W_try)
                    out = mlp.softmax(h2 @ mlp.W3)
                else:
                    out = mlp.softmax(layer_input @ W_try)

                ce = -np.mean(np.sum(y_true * np.log(out + 1e-9), axis=1))
                reg = WEIGHT_DECAY * (
                    np.sum(mlp.W1**2) + np.sum(mlp.W2**2) + np.sum(mlp.W3**2)
                )
                losses.append(ce + reg)

            losses = np.array(losses)
            best = np.min(losses)
            marked = np.where(losses == best)[0].tolist()
            idx = quantum_min(int(np.ceil(np.log2(len(candidates)))), marked, len(candidates), shots)

            W[i, j] = candidates[idx]

    return W


# =========================
# Data
# =========================
x, y = load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

x_train = (x_train - x_train.mean(0)) / x_train.std(0)
x_test = (x_test - x_test.mean(0)) / x_test.std(0)

enc = OneHotEncoder(sparse_output=False)
y_train_1h = enc.fit_transform(y_train.reshape(-1, 1))
y_test_1h = enc.transform(y_test.reshape(-1, 1))





# =========================
# ADAM baseline (2 hidden layers)
# =========================
def adam_baseline_train_losses(
    num_epochs,
    lr=1e-2,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):
    # identical init
    W1 = W1_INIT.copy()
    W2 = W2_INIT.copy()
    W3 = W3_INIT.copy()

    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)

    train_losses, test_losses = [], []

    # ---- Epoch 0 (pre-training) ----
    h1 = np.tanh(x_train @ W1)
    h2 = np.tanh(h1 @ W2)
    probs = np.exp(h2 @ W3 - np.max(h2 @ W3, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    ce = -np.mean(np.sum(y_train_1h * np.log(probs + 1e-9), axis=1))
    reg = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    train_losses.append(float(ce+reg))

    h1 = np.tanh(x_test @ W1)
    h2 = np.tanh(h1 @ W2)
    probs = np.exp(h2 @ W3 - np.max(h2 @ W3, axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    ce = -np.mean(np.sum(y_test_1h * np.log(probs + 1e-9), axis=1))
    reg = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    test_losses.append(float(ce+reg))

    # ---- Training ----
    for t in range(1, num_epochs + 1):
        # forward
        h1 = np.tanh(x_train @ W1)
        h2 = np.tanh(h1 @ W2)
        logits = h2 @ W3
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        N = x_train.shape[0]
        dZ3 = (probs - y_train_1h) / N

        gW3 = h2.T @ dZ3 + 2 * WEIGHT_DECAY * W3
        dH2 = dZ3 @ W3.T
        dZ2 = dH2 * (1 - h2**2)
        gW2 = h1.T @ dZ2 + 2 * WEIGHT_DECAY * W2
        dH1 = dZ2 @ W2.T
        dZ1 = dH1 * (1 - h1**2)
        gW1 = x_train.T @ dZ1 + 2 * WEIGHT_DECAY * W1

        # ADAM updates (bias-corrected)
        for W, g, m, v in [
            (W1, gW1, mW1, vW1),
            (W2, gW2, mW2, vW2),
            (W3, gW3, mW3, vW3),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g**2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            W -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # ---- losses ----
        h1 = np.tanh(x_train @ W1)
        h2 = np.tanh(h1 @ W2)
        probs = np.exp(h2 @ W3 - np.max(h2 @ W3, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        ce = -np.mean(np.sum(y_train_1h * np.log(probs + 1e-9), axis=1))
        reg = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        train_losses.append(float(ce + reg))

        h1 = np.tanh(x_test @ W1)
        h2 = np.tanh(h1 @ W2)
        probs = np.exp(h2 @ W3 - np.max(h2 @ W3, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        ce = -np.mean(np.sum(y_test_1h * np.log(probs + 1e-9), axis=1))
        reg = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        test_losses.append(float(ce + reg))

    return train_losses, test_losses

adam_train_loss_all, adam_test_loss_all = adam_baseline_train_losses(NUM_EPOCHS)


# =========================
# Grover training
# =========================
mlp_g = MLP()

grover_train_loss = []
grover_test_loss = []
gtr_a = []
gte_a = []

# Epoch 0
grover_train_loss.append(adam_train_loss_all[0])
grover_test_loss.append(adam_test_loss_all[0])

s1 = s2 = s3 = INIT_SEARCH_STD

for epoch in range(NUM_EPOCHS):
    print(f"\n=== Grover Epoch {epoch+1} ===")

    h1, h2, _ = mlp_g.forward(x_train)

    mlp_g.W1 = optimize_layer(mlp_g, 1, x_train, y_train_1h, mlp_g.W1, RES_W1_SF, s1, SHOTS)
    h1, h2, _ = mlp_g.forward(x_train)

    mlp_g.W2 = optimize_layer(mlp_g, 2, h1, y_train_1h, mlp_g.W2, RES_W2_SF, s2, SHOTS)
    h1, h2, _ = mlp_g.forward(x_train)

    mlp_g.W3 = optimize_layer(mlp_g, 3, h2, y_train_1h, mlp_g.W3, RES_W3_SF, s3, SHOTS)

    _, _, out_tr = mlp_g.forward(x_train)
    _, _, out_te = mlp_g.forward(x_test)

    grover_train_loss.append(-np.mean(np.sum(y_train_1h * np.log(out_tr + 1e-9), axis=1)))
    grover_test_loss.append(-np.mean(np.sum(y_test_1h * np.log(out_te + 1e-9), axis=1)))

    gtr_a.append((np.argmax(out_tr,1)==y_train).mean()*100)
    gte_a.append((np.argmax(out_te,1)==y_test).mean()*100)

    print(f"Train acc: {(np.argmax(out_tr,1)==y_train).mean()*100:.1f}% | "
          f"Test acc: {(np.argmax(out_te,1)==y_test).mean()*100:.1f}%")

# =========================
# Plot
# =========================
epochs_axis = np.arange(0, NUM_EPOCHS + 1)

print("epochs_axis:", len(epochs_axis))
print("grover_train:", len(grover_train_loss))
print("grover_test:", len(grover_test_loss))
print("adam_train:", len(adam_train_loss_all))
print("adam_test:", len(adam_test_loss_all))

print(gtr_a,gte_a)

plt.figure(figsize=(9, 5))

# ADAM
plt.plot(epochs_axis, adam_train_loss_all, marker='o', linestyle='-', linewidth=2, markersize=6, label="ADAM training loss")
plt.plot(epochs_axis, adam_test_loss_all, marker='o', linestyle='--', linewidth=2, markersize=6, label="ADAM test loss")

# Grover
plt.plot(epochs_axis, grover_train_loss, marker='o', linestyle='-', linewidth=2, markersize=6, label="Grover training loss")
plt.plot(epochs_axis, grover_test_loss, marker='o', linestyle='--', linewidth=2, markersize=6, label="Grover test loss")

plt.xlabel("Epoch (0 = pre-training)", fontsize=11)
plt.ylabel("Loss (cross-entropy + L2)", fontsize=11)
plt.title("Grover vs ADAM — identical initial weights verified", fontsize=13)

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


