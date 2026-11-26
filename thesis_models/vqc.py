"""Variational quantum circuits of the thesis, re-implemented as plain PennyLane.

Chapter 4 -- the original 1P1Q classifier ("one particle, one qubit"):
    10 qubits, 1 layer. Each wire w encodes particle w as RY(s pT eta) RX(s pT phi),
    a CNOT ring w -> (w+1) mod n entangles, and a trainable Rot(0, ry, rz) RX(rx) per wire
    follows. Readout <Z_0>; score = <Z_0> + b (MSE-trained, no readout coefficient).
    The 32 weights are one flat vector: w[3w + 0, 1, 2] = (rz, ry, rx) of wire w,
    w[-2] = scale parameter a with s = 2 pi sigmoid(a) + 1, w[-1] = bias b.

Chapter 5 -- data re-uploading, multi-qubit readouts and the R_YY(lambda_ij dR_ij) circuit:
    every layer l = 0..L-1 applies encode -> entangle -> rotate on the SAME particles
    (re-uploading), then one readout is measured.
      circuit "1P1Q": encode RY(s pT eta), RX(s pT phi); entangle CNOT ring.
      circuit "R_YY": encode RX(s pT); entangle IsingYY(lambda_ij dR_ij) on every pair i<j,
                      dR_ij = sqrt(d_eta^2 + d_phi^2 + 1e-12) from the scaled inputs,
                      lambda_ij trainable per pair and shared by all layers.
      rotate: Rot(0, ry, rz) then RX(rx) per wire (time order R_Y -> R_Z -> R_X),
              with (rz, ry, rx) = rot[l, w, 0..2].
      readout "z0": c_0 <Z_0>
              "z" : sum_i c_i <Z_i>
              "zz": sum_i c_i <Z_i> + sum_{i<j} c_ij <Z_i Z_j>
      score f = readout + b (a logit for BCE-trained models, a raw score for MSE).
    Parameter names are those of the saved weights: rot (L, n, 3), scale_factor, bias,
    hamiltonian_coeffs (1 or n), pairwise_coeffs (n(n-1)/2), dr_scale (n(n-1)/2).

Inputs are arrays x of shape (n_jets, n_particles >= n_qubits, 3), columns (eta, phi, pT)
after the readers' scaling: d_eta, d_phi of the pT-ordered constituents mapped from
[-R, R] onto [-pi, pi - 1e-4] (R = 0.8 for JetClass, 1.0 for JetsGame), pT divided by the
jet pT. Wire w reads particle w. The circuits are evaluated with parameter broadcasting
over the jets on default.qubit; shots=None gives exact (analytic) expectation values.

Numerical conventions follow the training code: the inputs are float32 (as the data loaders
yield them); encoding angles are formed in float64, while dR_ij is formed in the input
precision (float32, dr_precision="input") and then multiplied by the float64 lambda_ij.
The trained R_YY models ran this float32 step on a GPU, whose last-bit rounding of dR_ij
differs from NumPy's; dr_precision="float64" forms dR_ij exactly instead. Either choice
reproduces the stored R_YY scores to ~1e-7..1e-6 (up to ~2e-5 for deep BCE models): the
float32 resolution of dR_ij propagated through the circuit. The 1P1Q circuits have no float32
step and reproduce the stored scores to ~1e-14.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

ETA, PHI, PT = 0, 1, 2
CIRCUITS = ("1P1Q", "R_YY")
READOUTS = ("z0", "z", "zz")
READOUT_LABELS = {
    "z0": "<Z0>",
    "z": "sum_i c_i <Z_i>",
    "zz": "sum_i c_i <Z_i> + sum_ij c_ij <Z_i Z_j>",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def encoding_scale(a):
    """Trainable encoding scale s = 2 pi sigmoid(a) + 1, bounded to (1, 2 pi + 1)."""
    return 2.0 * np.pi * sigmoid(np.float64(a)) + 1.0


def pair_list(n_qubits):
    """All unordered wire pairs (i, j), i < j, in the order of the pairwise parameters."""
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def _device(n_qubits, shots, seed):
    """default.qubit; finite shots are attached to the QNode (or the device on old PennyLane)."""
    if shots is None or hasattr(qml, "set_shots"):
        return qml.device("default.qubit", wires=n_qubits, seed=seed)
    return qml.device("default.qubit", wires=n_qubits, shots=shots, seed=seed)


def _qnode(func, dev, shots):
    qnode = qml.QNode(func, dev, interface="numpy", diff_method=None)
    if shots is not None and hasattr(qml, "set_shots"):
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode


def _batched(fn, x, batch_size):
    x = np.asarray(x)
    out = [np.reshape(np.asarray(fn(x[i:i + batch_size])), (-1,))
           for i in range(0, len(x), batch_size)]
    return np.concatenate(out) if out else np.empty(0)


# --------------------------------------------------------------------------------------
# Chapter 5
# --------------------------------------------------------------------------------------
class Ch5VQC:
    """Chapter 5 circuit: `circuit` in CIRCUITS, `readout` in READOUTS.

    >>> model = Ch5VQC("1P1Q", "z0", n_qubits=7, n_layers=1)
    >>> f = model.scores(params, x)        # params: dict of the saved weights
    """

    def __init__(self, circuit, readout, n_qubits, n_layers, shots=None, seed=None,
                 dr_precision="input"):
        if circuit not in CIRCUITS:
            raise ValueError(f"circuit must be one of {CIRCUITS}, got {circuit!r}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.circuit, self.readout = circuit, readout
        self.n_qubits, self.n_layers = int(n_qubits), int(n_layers)
        self.wires = list(range(self.n_qubits))
        self.pairs = pair_list(self.n_qubits)
        if dr_precision not in ("input", "float64"):
            raise ValueError("dr_precision must be 'input' (as trained) or 'float64'")
        self.dr_precision = dr_precision
        self.shots = shots
        self.qnode = _qnode(self._circuit, _device(self.n_qubits, shots, seed), shots)

    # ---- parameters ------------------------------------------------------------------
    def param_shapes(self):
        n, L, n_pairs = self.n_qubits, self.n_layers, len(self.pairs)
        shapes = {"rot": (L, n, 3), "scale_factor": (), "bias": (),
                  "hamiltonian_coeffs": () if self.readout == "z0" else (n,)}
        if self.readout == "zz":
            shapes["pairwise_coeffs"] = (n_pairs,)
        if self.circuit == "R_YY":
            shapes["dr_scale"] = (n_pairs,)
        return shapes

    def n_params(self):
        """3nL + 2 + (1 | n | n + n(n-1)/2 readout coefficients) [+ n(n-1)/2 for R_YY]."""
        return int(sum(np.prod(s, dtype=int) for s in self.param_shapes().values()))

    def check_params(self, params):
        expected = self.param_shapes()
        if set(params) != set(expected):
            raise ValueError(f"parameter names {sorted(params)} != {sorted(expected)}")
        for name, shape in expected.items():
            if np.shape(params[name]) != shape:
                raise ValueError(f"{name}: shape {np.shape(params[name])}, expected {shape}")

    def random_params(self, seed=0):
        """Random parameters of the right shapes (for drawing / timing only)."""
        rng = np.random.default_rng(seed)
        return {k: rng.uniform(-np.pi, np.pi, s) if s else rng.uniform(-1, 1)
                for k, s in self.param_shapes().items()}

    # ---- circuit -----------------------------------------------------------------------
    def _encode(self, params, x):
        s = encoding_scale(params["scale_factor"])
        for w in self.wires:
            eta = x[:, w, ETA].astype(np.float64)
            phi = x[:, w, PHI].astype(np.float64)
            pt = x[:, w, PT].astype(np.float64)
            if self.circuit == "1P1Q":
                qml.RY(s * pt * eta, wires=w)
                qml.RX(s * pt * phi, wires=w)
            else:
                qml.RX(s * pt, wires=w)

    def _entangle(self, params, x):
        if self.circuit == "1P1Q":
            for w in self.wires:
                qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
            return
        lam = np.asarray(params["dr_scale"], dtype=np.float64)
        if self.dr_precision == "float64":
            x = x.astype(np.float64)
        for k, (i, j) in enumerate(self.pairs):
            d_eta = x[:, i, ETA] - x[:, j, ETA]          # input precision, as in training
            d_phi = x[:, i, PHI] - x[:, j, PHI]
            delta_r = np.sqrt(d_eta * d_eta + d_phi * d_phi + 1e-12)
            qml.IsingYY(lam[k] * delta_r.astype(np.float64), wires=[i, j])

    def _rotate(self, params, layer):
        rot = np.asarray(params["rot"], dtype=np.float64)[layer]
        for w in self.wires:
            rz, ry, rx = rot[w]
            qml.Rot(0.0, ry, rz, wires=w)   # = RZ(rz) RY(ry) RZ(0): R_Y first, then R_Z
            qml.RX(rx, wires=w)

    def _observable(self, params):
        c = np.asarray(params["hamiltonian_coeffs"], dtype=np.float64)
        if self.readout == "z0":
            return qml.Hamiltonian(np.reshape(c, (1,)), [qml.PauliZ(0)])
        coeffs, obs = list(c), [qml.PauliZ(w) for w in self.wires]
        if self.readout == "zz":
            coeffs += list(np.asarray(params["pairwise_coeffs"], dtype=np.float64))
            obs += [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in self.pairs]
        return qml.Hamiltonian(np.array(coeffs), obs)

    def _circuit(self, params, x):
        for layer in range(self.n_layers):
            self._encode(params, x)
            self._entangle(params, x)
            self._rotate(params, layer)
        return qml.expval(self._observable(params))

    # ---- evaluation --------------------------------------------------------------------
    def _x(self, x):
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[None]
        if x.shape[1] < self.n_qubits:
            raise ValueError(f"need >= {self.n_qubits} particles per jet, got {x.shape[1]}")
        return x[:, :self.n_qubits, :]

    def readout_values(self, params, x, batch_size=1000):
        """Readout expectation (without bias) per jet."""
        self.check_params(params)
        return _batched(lambda xb: self.qnode(params, xb), self._x(x), batch_size)

    def scores(self, params, x, batch_size=1000):
        """Classifier score f = readout + b per jet."""
        return self.readout_values(params, x, batch_size) + float(params["bias"])

    def draw(self, params, x, **kwargs):
        return qml.draw(self.qnode, **kwargs)(params, self._x(x)[:1])

    def __repr__(self):
        return (f"Ch5VQC({self.circuit}, readout={READOUT_LABELS[self.readout]}, "
                f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
                f"n_params={self.n_params()}, shots={self.shots})")


def ch5_n_params(circuit, readout, n_qubits, n_layers):
    n_pairs = n_qubits * (n_qubits - 1) // 2
    n_readout = {"z0": 1, "z": n_qubits, "zz": n_qubits + n_pairs}[readout]
    return 3 * n_qubits * n_layers + 2 + n_readout + (n_pairs if circuit == "R_YY" else 0)


# --------------------------------------------------------------------------------------
# Chapter 4
# --------------------------------------------------------------------------------------
class Ch4VQC:
    """Chapter 4 1P1Q classifier (frozen 1P1Q code, `_vqc_circuit`), flat weight vector.

    Layer l would read particles l*n .. l*n + n - 1 (no re-uploading); the thesis models
    have one layer. Thesis scores were estimated with 10^4 shots; shots=None is exact.
    """

    def __init__(self, n_qubits=10, n_layers=1, shots=None, seed=None):
        self.n_qubits, self.n_layers = int(n_qubits), int(n_layers)
        self.wires = list(range(self.n_qubits))
        self.shots = shots
        self.qnode = _qnode(self._circuit, _device(self.n_qubits, shots, seed), shots)

    def n_params(self):
        return 3 * self.n_qubits * self.n_layers + 2

    def _circuit(self, w, x):
        n = self.n_qubits
        s = encoding_scale(w[-2])
        for layer in range(self.n_layers):
            for q in self.wires:
                p = q + layer * n
                eta = x[:, p, ETA].astype(np.float64)
                phi = x[:, p, PHI].astype(np.float64)
                pt = x[:, p, PT].astype(np.float64)
                qml.RY(s * pt * eta, wires=q)
                qml.RX(s * pt * phi, wires=q)
            for q in self.wires:
                qml.CNOT(wires=[q, (q + 1) % n])
            start = 3 * layer * n
            rot_z = w[start + 0:start + 3 * n:3]
            rot_y = w[start + 1:start + 3 * n:3]
            rot_x = w[start + 2:start + 3 * n:3]
            for rz, ry, rx, q in zip(rot_z, rot_y, rot_x, self.wires):
                qml.Rot(0.0, ry, rz, wires=q)
                qml.RX(rx, wires=q)
        return qml.expval(qml.PauliZ(0))

    def _check(self, w, x):
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.n_params(),):
            raise ValueError(f"weights: shape {w.shape}, expected ({self.n_params()},)")
        x = np.asarray(x)
        need = self.n_qubits * self.n_layers
        if x.shape[1] < need:
            raise ValueError(f"need >= {need} particles per jet, got {x.shape[1]}")
        return w, x[:, :need, :]

    def scores(self, w, x, batch_size=1000):
        """Score <Z_0> + b per jet."""
        w, x = self._check(w, x)
        return _batched(lambda xb: self.qnode(w, xb), x, batch_size) + w[-1]

    def draw(self, w, x, **kwargs):
        w, x = self._check(w, x)
        return qml.draw(self.qnode, **kwargs)(w, x[:1])

    def __repr__(self):
        return (f"Ch4VQC(1P1Q, readout=<Z0>, n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, n_params={self.n_params()}, shots={self.shots})")
