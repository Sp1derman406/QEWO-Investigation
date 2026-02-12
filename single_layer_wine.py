#single layer wine
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from math import pi, floor, sqrt


# =========================
# SPEED / TEST CONFIG
# =========================
# Keep algorithm identical; only reduce budget.


#default = 10, 32, 16, 32
NUM_EPOCHS = 10        # paper uses 10; start with 2–3 to sanity-check runtime/shape
# 10 = 1 hr (with default)
# 2 = 15m
#
HIDDEN_SIZE = 4
# 5 = 9m
# 62 = 33m (3 epoch)

RES_HIDDEN = 3          # paper/main.py uses 32
# 8 = 40m
# 64 = 4h

RES_OUTPUT = 3             # paper/main.py uses 64


SHOTS = 256               # paper/main.py uses 1024; 256 is fine for a quick test

DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-3

INIT_SEARCH_STD = 0.5
MIN_SEARCH_STD = 0.1
MAX_SEARCH_STD = 2.0

TOL_HIDDEN = 0.05
TOL_OUTPUT = 0.10

PRINT_INNER_PROGRESS = False   # set True if want per-weight progress for debugging

hw = np.random.uniform(-1, 1, (13, HIDDEN_SIZE))
ow = np.random.uniform(-1, 1, (HIDDEN_SIZE, 3))
global hw
global ow


# =========================
# Grover components (logic from main.py)
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

def quantum_min(num_qubits, marked_states, N, shots=1024):
    M = len(marked_states)
    if M == 0:
        raise ValueError("No marked states for Grover search.")
    k = floor((pi / 4) * sqrt(N / M))
    if k < 1:
        k = 1

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

    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    if not counts:
        raise ValueError("No results obtained from simulation.")
    max_state = max(counts, key=counts.get)
    chosen_index = int(max_state, 2)
    if chosen_index >= N:
        # fallback (rare): clamp to valid range
        chosen_index = chosen_index % N
    return chosen_index


# =========================
# MLP + forward (logic froms main.py)
# =========================
class MLP:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        np.random.seed(42)
        self.weights_1 = hw
        self.weights_2 = ow
        self.dropout_rate = dropout_rate

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x, training=True):
        hidden = np.tanh(np.dot(x, self.weights_1))
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout_rate).astype(float)
            hidden *= mask
            hidden /= (1.0 - self.dropout_rate)
        output = self.softmax(np.dot(hidden, self.weights_2))
        return hidden, output

def forward_with_custom_hidden(mlp, x, custom_hidden_weights, output_weights, training=False):
    hidden = np.tanh(np.dot(x, custom_hidden_weights))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden.shape) > mlp.dropout_rate).astype(float)
        hidden *= mask
        hidden /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden, output_weights))
    return hidden, output

def forward_with_custom_output(mlp, hidden_input, custom_output_weights, training=False):
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden_input.shape) > mlp.dropout_rate).astype(float)
        hidden_input *= mask
        hidden_input /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden_input, custom_output_weights))
    return hidden_input, output


# =========================
# Weight optimizer (based upon main.py, plus optional progress print out)
# =========================
def optimize_weights_with_grover(
    mlp,
    layer_input,
    y_true,
    weights,
    layer_type,
    resolution,
    search_std_factor,
    weight_decay,
    tol_ratio,
    shots
):
    optimized_weights = weights.copy()
    std_weight = np.std(weights)
    if std_weight < 1e-12:
        std_weight = 1.0

    total_params = weights.shape[0] * weights.shape[1]
    done = 0

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            current_weight = weights[i, j]
            low  = current_weight - search_std_factor * std_weight
            high = current_weight + search_std_factor * std_weight
            candidates = np.linspace(low, high, resolution)

            losses = []
            for cand in candidates:
                temp_weights = optimized_weights.copy()
                temp_weights[i, j] = cand

                if layer_type == "hidden":
                    _, output = forward_with_custom_hidden(mlp, layer_input, temp_weights, mlp.weights_2, training=False)
                    w1_s = np.sum(temp_weights**2)
                    w2_s = np.sum(mlp.weights_2**2)
                else:
                    _, output = forward_with_custom_output(mlp, layer_input, temp_weights, training=False)
                    w1_s = np.sum(mlp.weights_1**2)
                    w2_s = np.sum(temp_weights**2)

                ce_loss = -np.mean(np.sum(y_true * np.log(output + 1e-9), axis=1))
                loss = ce_loss + weight_decay * (w1_s + w2_s)
                losses.append(loss)

            losses = np.asarray(losses, dtype=float)
            min_loss = float(np.min(losses))
            tol = tol_ratio * min_loss
            marked_states = [idx for idx, L in enumerate(losses) if L <= min_loss + tol]

            N = len(candidates)
            num_qubits = int(np.ceil(np.log2(N)))
            chosen_index = quantum_min(num_qubits, marked_states, N, shots=shots)

            proposed_weight = candidates[chosen_index]
            proposed_weights = optimized_weights.copy()
            proposed_weights[i, j] = proposed_weight

            # accept-if-better (same logic as main.py)
            if layer_type == "hidden":
                _, new_output = forward_with_custom_hidden(mlp, layer_input, proposed_weights, mlp.weights_2, training=False)
                new_ce = -np.mean(np.sum(y_true * np.log(new_output + 1e-9), axis=1))
                new_loss = new_ce + weight_decay * (np.sum(proposed_weights**2) + np.sum(mlp.weights_2**2))

                _, cur_output = forward_with_custom_hidden(mlp, layer_input, optimized_weights, mlp.weights_2, training=False)
                cur_ce = -np.mean(np.sum(y_true * np.log(cur_output + 1e-9), axis=1))
                cur_loss = cur_ce + weight_decay * (np.sum(optimized_weights**2) + np.sum(mlp.weights_2**2))
            else:
                _, new_output = forward_with_custom_output(mlp, layer_input, proposed_weights, training=False)
                new_ce = -np.mean(np.sum(y_true * np.log(new_output + 1e-9), axis=1))
                new_loss = new_ce + weight_decay * (np.sum(mlp.weights_1**2) + np.sum(proposed_weights**2))

                _, cur_output = forward_with_custom_output(mlp, layer_input, optimized_weights, training=False)
                cur_ce = -np.mean(np.sum(y_true * np.log(cur_output + 1e-9), axis=1))
                cur_loss = cur_ce + weight_decay * (np.sum(mlp.weights_1**2) + np.sum(optimized_weights**2))

            if new_loss < cur_loss:
                optimized_weights[i, j] = proposed_weight

            done += 1
            if PRINT_INNER_PROGRESS and (done % 50 == 0):
                print(f"    {layer_type}: optimized {done}/{total_params} weights...")

    return optimized_weights


# =========================
# Data prep (as main.py)
# =========================
data = load_wine()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test  = (x_test  - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

encoder = OneHotEncoder(sparse_output=False)
y_train_1hot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_1hot  = encoder.transform(y_test.reshape(-1, 1))


# =========================
# Grover training loop (main.py logic) + per-epoch timing
# =========================
mlp_g = MLP(input_size=13, hidden_size=HIDDEN_SIZE, output_size=3, dropout_rate=DROPOUT_RATE)

search_std_factor_hidden = INIT_SEARCH_STD
search_std_factor_output = INIT_SEARCH_STD

grover_train_loss_all, grover_test_loss_all = [], []
grover_train_acc_all,  grover_test_acc_all  = [], []

#############################################################
_, y_train_pred = mlp_g.forward(x_train, training=False)
train_ce = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred + 1e-9), axis=1))
train_reg = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
train_loss = train_ce + train_reg
train_acc = (np.argmax(y_train_pred, axis=1) == y_train).mean() * 100

_, y_test_pred = mlp_g.forward(x_test, training=False)
test_ce = -np.mean(np.sum(y_test_1hot * np.log(y_test_pred + 1e-9), axis=1))
test_reg = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
test_loss = test_ce + test_reg
test_acc = (np.argmax(y_test_pred, axis=1) == y_test).mean() * 100

grover_train_loss_all.append(train_loss)
grover_test_loss_all.append(test_loss)
grover_train_acc_all.append(train_acc)
grover_test_acc_all.append(test_acc)
###############################################################

t0_all = time.time()

for epoch in range(NUM_EPOCHS):
    t0_epoch = time.time()
    print(f"\n=== Grover Epoch {epoch+1}/{NUM_EPOCHS} ===")
    print(f"  search_std_factor_hidden={search_std_factor_hidden:.4f} | search_std_factor_output={search_std_factor_output:.4f}")

    # ---- Hidden layer before ----
    hidden_before, _ = mlp_g.forward(x_train, training=False)
    train_probs_before = mlp_g.softmax(np.dot(hidden_before, mlp_g.weights_2))
    ce_before_h = -np.mean(np.sum(y_train_1hot * np.log(train_probs_before + 1e-9), axis=1))
    reg_before_h = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
    loss_before_h = ce_before_h + reg_before_h

    # ---- Optimize W1 ----
    mlp_g.weights_1 = optimize_weights_with_grover(
        mlp=mlp_g,
        layer_input=x_train,
        y_true=y_train_1hot,
        weights=mlp_g.weights_1,
        layer_type="hidden",
        resolution=RES_HIDDEN,
        search_std_factor=search_std_factor_hidden,
        weight_decay=WEIGHT_DECAY,
        tol_ratio=TOL_HIDDEN,
        shots=SHOTS
    )

    # ---- Hidden layer after; update search_std_factor_hidden ----
    hidden_after, _ = mlp_g.forward(x_train, training=False)
    train_probs_after_h = mlp_g.softmax(np.dot(hidden_after, mlp_g.weights_2))
    ce_after_h = -np.mean(np.sum(y_train_1hot * np.log(train_probs_after_h + 1e-9), axis=1))
    reg_after_h = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
    loss_after_h = ce_after_h + reg_after_h

    if loss_after_h < loss_before_h:
        search_std_factor_hidden = max(search_std_factor_hidden * 0.95, MIN_SEARCH_STD)
    else:
        search_std_factor_hidden = min(search_std_factor_hidden * 1.05, MAX_SEARCH_STD)

    # ---- Output layer optimization uses hidden activations as input ----
    hidden_out_input, _ = mlp_g.forward(x_train, training=False)
    loss_before_o = loss_after_h

    # ---- Optimize W2 ----
    mlp_g.weights_2 = optimize_weights_with_grover(
        mlp=mlp_g,
        layer_input=hidden_out_input,
        y_true=y_train_1hot,
        weights=mlp_g.weights_2,
        layer_type="output",
        resolution=RES_OUTPUT,
        search_std_factor=search_std_factor_output,
        weight_decay=WEIGHT_DECAY,
        tol_ratio=TOL_OUTPUT,
        shots=SHOTS
    )

    # ---- Output layer after; update search_std_factor_output ----
    _, y_train_pred_after_o = mlp_g.forward(x_train, training=False)
    ce_after_o = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred_after_o + 1e-9), axis=1))
    reg_after_o = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
    loss_after_o = ce_after_o + reg_after_o

    if loss_after_o < loss_before_o:
        search_std_factor_output = max(search_std_factor_output * 0.95, MIN_SEARCH_STD)
    else:
        search_std_factor_output = min(search_std_factor_output * 1.05, MAX_SEARCH_STD)

    # ---- Final epoch metrics ----
    _, y_train_pred = mlp_g.forward(x_train, training=False)
    train_ce = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred + 1e-9), axis=1))
    train_reg = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
    train_loss = train_ce + train_reg
    train_acc = (np.argmax(y_train_pred, axis=1) == y_train).mean() * 100

    _, y_test_pred = mlp_g.forward(x_test, training=False)
    test_ce = -np.mean(np.sum(y_test_1hot * np.log(y_test_pred + 1e-9), axis=1))
    test_reg = WEIGHT_DECAY * (np.sum(mlp_g.weights_1**2) + np.sum(mlp_g.weights_2**2))
    test_loss = test_ce + test_reg
    test_acc = (np.argmax(y_test_pred, axis=1) == y_test).mean() * 100

    grover_train_loss_all.append(train_loss)
    grover_test_loss_all.append(test_loss)
    grover_train_acc_all.append(train_acc)
    grover_test_acc_all.append(test_acc)

    t1_epoch = time.time()
    print(f"  Epoch metrics: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
)
    print(f"  Epoch time: {t1_epoch - t0_epoch:.1f} sec | Total elapsed: {t1_epoch - t0_all:.1f} sec")


# =========================
# Simple ADAM baseline (unchanged algorithm for Grover; baseline for Figure-2 plot)
# =========================
def adam_baseline_train_losses(num_epochs, hidden_size, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    rng = np.random.default_rng(42)
    W1 = hw
    W2 = ow

    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)

    tr_losses, te_losses = [], []

    ############################################################
    hidden_tr = np.tanh(x_train @ W1)
    probs_tr = np.exp(hidden_tr @ W2 - np.max(hidden_tr @ W2, axis=1, keepdims=True))
    probs_tr = probs_tr / np.sum(probs_tr, axis=1, keepdims=True)
    ce_tr = -np.mean(np.sum(y_train_1hot * np.log(probs_tr + 1e-9), axis=1))
    reg_tr = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2))
    tr_losses.append(float(ce_tr + reg_tr))

    hidden_te = np.tanh(x_test @ W1)
    probs_te = np.exp(hidden_te @ W2 - np.max(hidden_te @ W2, axis=1, keepdims=True))
    probs_te = probs_te / np.sum(probs_te, axis=1, keepdims=True)
    ce_te = -np.mean(np.sum(y_test_1hot * np.log(probs_te + 1e-9), axis=1))
    reg_te = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2))
    te_losses.append(float(ce_te + reg_te))
    ##############################################################

    for t in range(1, num_epochs + 1):
        hidden = np.tanh(x_train @ W1)
        logits = hidden @ W2
        exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        N = x_train.shape[0]
        dlogits = (probs - y_train_1hot) / N

        gW2 = hidden.T @ dlogits + 2 * WEIGHT_DECAY * W2
        dHidden = dlogits @ W2.T
        dZ = dHidden * (1.0 - hidden**2)
        gW1 = x_train.T @ dZ + 2 * WEIGHT_DECAY * W1

        mW1 = beta1 * mW1 + (1 - beta1) * gW1
        vW1 = beta2 * vW1 + (1 - beta2) * (gW1**2)
        mW2 = beta1 * mW2 + (1 - beta1) * gW2
        vW2 = beta2 * vW2 + (1 - beta2) * (gW2**2)

        mW1_hat = mW1 / (1 - beta1**t); vW1_hat = vW1 / (1 - beta2**t)
        mW2_hat = mW2 / (1 - beta1**t); vW2_hat = vW2 / (1 - beta2**t)

        W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)
        W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)

        # losses (CE+L2)
        hidden_tr = np.tanh(x_train @ W1)
        probs_tr = np.exp(hidden_tr @ W2 - np.max(hidden_tr @ W2, axis=1, keepdims=True))
        probs_tr = probs_tr / np.sum(probs_tr, axis=1, keepdims=True)
        ce_tr = -np.mean(np.sum(y_train_1hot * np.log(probs_tr + 1e-9), axis=1))
        reg_tr = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2))
        tr_losses.append(float(ce_tr + reg_tr))

        hidden_te = np.tanh(x_test @ W1)
        probs_te = np.exp(hidden_te @ W2 - np.max(hidden_te @ W2, axis=1, keepdims=True))
        probs_te = probs_te / np.sum(probs_te, axis=1, keepdims=True)
        ce_te = -np.mean(np.sum(y_test_1hot * np.log(probs_te + 1e-9), axis=1))
        reg_te = WEIGHT_DECAY * (np.sum(W1**2) + np.sum(W2**2))
        te_losses.append(float(ce_te + reg_te))

    return tr_losses, te_losses

adam_train_loss_all, adam_test_loss_all = adam_baseline_train_losses(NUM_EPOCHS, HIDDEN_SIZE, lr=1e-2)


# =========================
# Figure-2 style plot
# =========================

print(adam_train_loss_all)
print(adam_test_loss_all)
print(grover_train_loss_all)
print(grover_test_loss_all)

epochs_axis = np.arange(0, NUM_EPOCHS + 1) # Changed from 1 to 0 to match loss list length

plt.figure(figsize=(9, 5))
plt.grid(True, linestyle=':', alpha=0.6)
plt.plot(epochs_axis, adam_train_loss_all, marker='o', label='ADAM training loss')
plt.plot(epochs_axis, adam_test_loss_all, linestyle='--', label='ADAM test loss')
plt.plot(epochs_axis, grover_train_loss_all, marker='o', label='Grover training loss')
plt.plot(epochs_axis, grover_test_loss_all, linestyle='--', label='Grover test loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (cross-entropy + L2)")
plt.title("Figure-2 Style Comparison: Grover vs ADAM (main.py-faithful, faster budget)")
plt.legend()
plt.show()
