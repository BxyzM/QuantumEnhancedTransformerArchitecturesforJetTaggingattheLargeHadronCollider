"""DimeNet++ and QDimeNet++ for the QM9 HOMO-LUMO gap (thesis QM9 chapter, run006).

Self-contained re-implementation of the two models used in the thesis:

* ``DimeNetPP``  -- PyG DimeNet++ on heavy atoms (the classical baseline).
* ``QDimeNetPP`` -- the same backbone plus one QFIM edge-rescale adapter
  (``DimeNetPPQFIMLegacy`` in the qm9-qinn repository, model type
  ``dimenet_pp_qfim``). For every directed edge j -> i the 6x6 QFIM block
  Q[j, i] of the two atoms' qubits is encoded to a scalar s in (-1, 1), and the
  edge state is rescaled  m_ji <- m_ji * (1 + beta * alpha * s)  after the
  embedding block and after each of the 4 interaction blocks. Head, scale MLP
  and the scalar gate alpha are shared by all 5 injection points; beta = 1.

Only the DimeNet++ building blocks (basis layers, embedding, interaction and
output blocks) are taken from ``torch_geometric``; the graph construction
(radius graph, triplets) is done here in plain PyTorch, so neither
``torch_cluster`` nor ``torch_sparse`` is needed. Parameter names match the
thesis checkpoints, so both state dicts load with ``strict=True``.

Inputs per molecule (heavy atoms only, in the order stored in the QM9 HDF5
file): atomic numbers ``z``, positions ``pos`` in Angstrom and, for QDimeNet++,
the QFIM tensor ``qfim`` of shape (10, 10, 6, 6) = (qubit, qubit, param, param),
i.e. the 60x60 Rot-gate block of the stored metric tensor reordered with
``reshape(6, 10, 6, 10).transpose(1, 3, 0, 2)``. Both models predict the
standardized gap; multiply by ``target_std`` and add ``target_mean`` for eV.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.utils import scatter

# Hyperparameters of both run006 models (qm9_dimenet_pp_heavy[_qfim]_pat30_delta025_run006_seed42.yaml).
DIMENET_PP_HPARAMS = dict(
    hidden_channels=128,
    out_channels=1,
    num_blocks=4,
    int_emb_size=64,
    basis_emb_size=8,
    out_emb_channels=256,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
    max_num_neighbors=32,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=2,
    num_output_layers=3,
    act="swish",
    output_initializer="zeros",
)
QFIM_N_QUBITS = 10   # qubits of the bioQINN circuit (first 10 atom rows: heavy atoms first)
QFIM_PARAM_DIM = 6   # Rot-gate parameters per qubit: 2 layers x Rot(phi, theta, omega)


# ---------------------------------------------------------------------------
# Graph construction (plain PyTorch)
# ---------------------------------------------------------------------------

def _expand_ranges(starts: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate arange(s, s + l) for every (s, l); also return the owning row of each value."""
    owner = torch.repeat_interleave(torch.arange(starts.numel(), device=starts.device), lengths)
    first = torch.cumsum(lengths, 0) - lengths
    values = starts[owner] + torch.arange(owner.numel(), device=starts.device) - first[owner]
    return owner, values


def radius_graph(pos: torch.Tensor, batch: torch.Tensor, cutoff: float,
                 max_num_neighbors: int = 32) -> torch.Tensor:
    """All directed edges j -> i between distinct atoms of the same molecule with |r_i - r_j| < cutoff.

    Returns ``edge_index`` (2, E) with row 0 = source j and row 1 = target i (PyG
    ``source_to_target``). ``batch`` must be sorted (atoms of a molecule contiguous).
    """
    counts = torch.bincount(batch)
    ptr = torch.cumsum(counts, 0) - counts
    tgt, src = _expand_ranges(ptr[batch], counts[batch])        # every intra-molecule pair
    dist = (pos[tgt] - pos[src]).norm(dim=-1)
    keep = (src != tgt) & (dist < cutoff)
    src, tgt = src[keep], tgt[keep]
    n_neigh = torch.bincount(tgt, minlength=pos.size(0))
    if n_neigh.numel() and int(n_neigh.max()) > max_num_neighbors:
        # Never happens for heavy-atom QM9 graphs (<= 9 atoms); refuse rather than guess the truncation rule.
        raise NotImplementedError(f"an atom has {int(n_neigh.max())} > {max_num_neighbors} neighbours")
    return torch.stack([src, tgt])


def triplets(edge_index: torch.Tensor, num_nodes: int):
    """DimeNet triplets k -> j -> i (k != i); same outputs as ``torch_geometric.nn.models.dimenet.triplets``."""
    row, col = edge_index                                        # j -> i
    order = torch.argsort(col, stable=True)                      # edges grouped by target node
    deg_in = torch.bincount(col, minlength=num_nodes)
    first_in = torch.cumsum(deg_in, 0) - deg_in
    # For edge e = (j -> i) take every incoming edge (k -> j) of its source j.
    idx_ji, slot = _expand_ranges(first_in[row], deg_in[row])
    idx_kj = order[slot]
    idx_i, idx_j, idx_k = col[idx_ji], row[idx_ji], row[idx_kj]
    mask = idx_i != idx_k
    return col, row, idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]


def gather_edge_qfim(qfim: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                     n_qubits: int = QFIM_N_QUBITS) -> torch.Tensor:
    """Per edge (j -> i): flattened block Q[mol, local(j), local(i)]; zeros if a local atom index >= n_qubits."""
    src, dst = edge_index
    counts = torch.bincount(batch)
    ptr = torch.cumsum(counts, 0) - counts
    local = torch.arange(batch.numel(), device=batch.device) - ptr[batch]
    li, lj = local[src], local[dst]
    in_budget = (li < n_qubits) & (lj < n_qubits)
    sub = qfim[batch[src], li.clamp(max=n_qubits - 1), lj.clamp(max=n_qubits - 1)]   # (E, pd, pd)
    sub = sub * in_budget.view(-1, 1, 1).to(sub.dtype)
    return sub.reshape(sub.shape[0], -1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QFIMHeadConv2d(nn.Module):
    """6x6 QFIM block -> 8-dim edge code: Conv2d(1->16) -> LN -> ReLU -> Conv2d(16->16) -> LN -> ReLU -> pool -> Linear -> LN."""

    def __init__(self, pd: int = QFIM_PARAM_DIM, out_dim: int = 8, conv_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.pd = pd
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm([conv_channels, pd, pd])
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm([conv_channels, pd, pd])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(conv_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        n = qfim_edge.shape[0]
        h = qfim_edge.view(n, 1, self.pd, self.pd)
        h = torch.relu(self.norm1(self.conv1(h)))
        h = torch.relu(self.norm2(self.conv2(h)))
        return self.out_norm(self.project(self.pool(h).view(n, -1)))


class DimeNetPP(nn.Module):
    """Classical baseline: PyG DimeNet++ on heavy atoms. ``forward(z, pos, batch)`` -> (B,) standardized gap."""

    def __init__(self, **hparams):
        super().__init__()
        self.model = DimeNetPlusPlus(**{**DIMENET_PP_HPARAMS, **hparams})

    def _geometry(self, z, pos, batch):
        m = self.model
        edge_index = radius_graph(pos, batch, m.cutoff, m.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        angle = torch.atan2(torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1), (pos_ij * pos_jk).sum(dim=-1))
        return edge_index, i, idx_kj, idx_ji, m.rbf(dist), m.sbf(dist, angle, idx_kj)

    def _run(self, z, pos, batch, rescale=None):
        m = self.model
        num_graphs = int(batch.max()) + 1
        edge_index, i, idx_kj, idx_ji, rbf, sbf = self._geometry(z, pos, batch)
        rescale = rescale(edge_index) if rescale is not None else (lambda e: e)
        x = rescale(m.emb(z, rbf, i, edge_index[0]))                    # edge states m_ji after embedding
        out = m.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(m.interaction_blocks, m.output_blocks[1:]):
            x = rescale(interaction_block(x, rbf, sbf, idx_kj, idx_ji))  # ... and after each interaction block
            out = out + output_block(x, rbf, i, num_nodes=pos.size(0))
        return scatter(out, batch, dim=0, dim_size=num_graphs, reduce="sum").view(-1)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self._run(z, pos, batch)


class QDimeNetPP(DimeNetPP):
    """DimeNet++ + shared QFIM edge rescale m_ji <- m_ji (1 + beta alpha s_ji). ``forward(z, pos, batch, qfim)``."""

    def __init__(self, qfim_out_dim: int = 8, qfim_residual_gate_init: float = 0.0,
                 qfim_rescale_beta: float = 1.0, **hparams):
        super().__init__(**hparams)
        self.qfim_head = QFIMHeadConv2d(QFIM_PARAM_DIM, qfim_out_dim)
        self.qfim_scale = nn.Sequential(
            nn.Linear(qfim_out_dim, max(4, qfim_out_dim)), nn.SiLU(),
            nn.Linear(max(4, qfim_out_dim), 1), nn.Tanh(),
        )
        self.qfim_residual_gate = nn.Parameter(torch.tensor(float(qfim_residual_gate_init)))   # alpha
        self.qfim_rescale_beta = float(qfim_rescale_beta)                                      # beta

    def edge_score(self, qfim: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """s_ji in (-1, 1), shape (E, 1). It depends only on the QFIM, so it is the same at all 5 injection points."""
        return self.qfim_scale(self.qfim_head(gather_edge_qfim(qfim, edge_index, batch)))

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, qfim: torch.Tensor) -> torch.Tensor:
        def make_rescale(edge_index):
            s = self.edge_score(qfim, edge_index, batch)
            return lambda e: e + self.qfim_rescale_beta * self.qfim_residual_gate * s * e
        return self._run(z, pos, batch, rescale=make_rescale)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def load_demo(path: str) -> dict:
    """Load data/qm9_test_demo.npz into a dict of numpy arrays (see tools/export_qm9.py for the keys)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def molecule_batches(demo: dict, batch_size: int = 128) -> Iterator[Tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (molecule slice, z, pos, batch, qfim) tensors for consecutive molecules of the demo set."""
    ptr = demo["atom_ptr"]
    n_mol = ptr.size - 1
    for start in range(0, n_mol, batch_size):
        stop = min(start + batch_size, n_mol)
        a, b = int(ptr[start]), int(ptr[stop])
        counts = torch.from_numpy(np.diff(ptr[start:stop + 1]))
        batch = torch.repeat_interleave(torch.arange(stop - start), counts)
        z = torch.from_numpy(demo["z"][a:b].astype(np.int64))
        pos = torch.from_numpy(demo["pos"][a:b])
        qfim = torch.from_numpy(demo["qfim"][start:stop])
        yield slice(start, stop), z, pos, batch, qfim
