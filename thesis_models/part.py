"""Particle Transformer (ParT) and QUIVER (ParT + per-particle QFIM tokens, "PConcat") for the thesis checkpoints.

Self-contained (torch only) copy of the code the thesis models were trained and evaluated with:

* ``weaver-core/weaver/nn/model/ParticleTransformer.py`` (the weaver-core copy vendored in the thesis
  repository, commit 3231e58): ``Embed``, ``PairEmbed``, ``Block``, ``SequenceTrimmer``, ``ParticleTransformer``;
* ``networks/example_ParticleTransformer.py``: ParT wrapper and ``get_model`` defaults;
* ``networks/QFI-augmented/example_ParticleTransformer_qfim_simple.py`` with
  ``qfim_concat_mode='post_embed_particle_concat'`` (QUIVER).

Only the code paths these models use are kept (no ONNX, no extra pair features, no G1 gating, no QFIM ablations,
no other fusion modes).  Module and parameter names are unchanged, so the thesis state dicts load with
``strict=True``; the forward passes are the same operations in the same order, so scores agree with weaver's to
floating-point round-off.

Inputs, as weaver feeds them (N jets, P = 10 particles, see the data configs):

    points           (N, 2, P)   part_deta, part_dphi             (not used by the network)
    features         (N, C, P)   C = 3 (Kin) or 17 (Full) standardized particle features
    lorentz_vectors  (N, 4, P)   part_px, part_py, part_pz, part_energy
    mask             (N, 1, P)   1 for real particles, 0 for padding
    qfim             (N, 90, P)  QUIVER only: token p = (qf0[p], ..., qf89[p])

The network returns logits for (QCD, signal); weaver's score is ``softmax(logits)[:, 1]``.
"""
import copy
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ParticleTransformer', 'ParTWrapper', 'QuiverWrapper', 'build_part', 'build_quiver', 'build_model',
           'load_model', 'predict', 'PF_INPUT_DIM', 'QFIM_INPUT_DIM']

PF_INPUT_DIM = {'kin': 3, 'full': 17}  # len(input_dicts['pf_features']) of the Kin / Full data configs
QFIM_INPUT_DIM = 90                    # qf0 ... qf89


# ----------------------------------------------------------------------------------------------------------------
# Pairwise Lorentz-vector features (ParT interaction matrix): ln kT, ln z, ln DeltaR, ln m^2
# ----------------------------------------------------------------------------------------------------------------

@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    m = torch.sqrt(to_m2(x, eps=eps))
    return torch.cat((pt, rapidity, phi, m), dim=1)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8):
    if num_outputs not in (1, 2, 3, 4):
        raise ValueError('only the 1- or 4-feature pair embedding is implemented here')
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    ptmin = torch.minimum(pti, ptj)
    lnkt = torch.log((ptmin * delta).clamp(min=eps))
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
    outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


# ----------------------------------------------------------------------------------------------------------------
# ParT building blocks
# ----------------------------------------------------------------------------------------------------------------

class SequenceTrimmer(nn.Module):
    """Drops trailing padded particles. In eval mode it only cuts the batch to its longest jet (after 5 calls)."""

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P); v: (N, 4, P) [px,py,pz,energy]; mask: (N, 1, P) -- real particle = 1, padded = 0
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    """BatchNorm over the input features, then [LayerNorm -> Linear -> GELU] per entry of ``dims``."""

    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    """Pairwise Lorentz-vector features -> per-head attention bias (weaver ``PairEmbed``, mode='sum', no extra pair inputs)."""

    def __init__(self, pairwise_lv_dim, dims, remove_self_pair=False, use_pre_activation_pair=True,
                 normalize_input=True, activation='gelu', eps=1e-8):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.remove_self_pair = remove_self_pair
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps)
        self.out_dim = dims[-1]

        input_dim = pairwise_lv_dim
        module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        if use_pre_activation_pair:
            module_list = module_list[:-1]
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        # x: (batch, 4, seq_len) Lorentz vectors; the features are symmetric, so only i >= j is computed
        batch_size, _, seq_len = x.size()
        i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0, device=x.device)
        with torch.no_grad():
            x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
            xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
            xj = x[:, :, j, i]
            x = self.pairwise_lv_fts(xi, xj)

        elements = self.embed(x)  # (batch, embed_dim, num_elements)

        y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
        y[:, :, i, j] = elements
        y[:, :, j, i] = elements
        return y


class Block(nn.Module):
    """ParT attention block (particle self-attention, or class attention when ``x_cls`` is given)."""

    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        # x: (seq_len, batch, embed_dim); x_cls: (1, batch, embed_dim); padding_mask: (batch, seq_len), padded = True/-inf
        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
        ) if pair_embed_dims is not None and pair_input_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.cls_token, std=.02)  # overwritten by the checkpoint

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P); v: (N, 4, P) [px,py,pz,energy]; mask: (N, 1, P) -- real particle = 1, padded = 0
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if v is not None and self.pair_embed is not None:
                attn_mask = self.pair_embed(v).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)
                # attn_mask is float: use the matching additive float padding mask
                padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output


# ----------------------------------------------------------------------------------------------------------------
# Thesis models
# ----------------------------------------------------------------------------------------------------------------

class ParTWrapper(nn.Module):
    """ParT baseline (networks/example_ParticleTransformer.py)."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


class QuiverWrapper(nn.Module):
    """QUIVER: ParT with one extra token per particle built from its 90 QFIM entries
    (networks/QFI-augmented/example_ParticleTransformer_qfim_simple.py, qfim_concat_mode='post_embed_particle_concat').

    features -> pf_embed   -> P particle tokens ─┐
                                                  ├─ concatenated along the sequence (2P tokens) -> ParT blocks
    qfim     -> qfim_embed -> P QFIM tokens     ─┘
    The ParT pair-interaction bias covers the P x P particle block and is zero-padded to 2P x 2P;
    no padding mask is applied to the 2P tokens (as in the trained model).
    """

    def __init__(self, pf_input_dim, qfim_input_dim=QFIM_INPUT_DIM, qfim_token_alpha=1.0, **kwargs) -> None:
        super().__init__()
        self.pf_input_dim = pf_input_dim
        self.qfim_input_dim = qfim_input_dim
        self.register_buffer('qfim_token_alpha', torch.tensor(float(qfim_token_alpha), dtype=torch.float32))

        part_kwargs = dict(kwargs)
        activation = part_kwargs.get('activation', 'gelu')
        embed_dims = part_kwargs.get('embed_dims', [128, 512, 128])
        trim = part_kwargs.get('trim', True)
        for_inference = part_kwargs.get('for_inference', False)

        # trimming and embedding happen here, outside the core ParT
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.qfim_embed = Embed(qfim_input_dim, list(embed_dims), activation=activation)

        part_kwargs['input_dim'] = embed_dims[-1]
        part_kwargs['embed_dims'] = []
        part_kwargs['trim'] = False
        self.mod = ParticleTransformer(**part_kwargs)

    def _forward_particle_concat(self, features, lorentz_vectors):
        # features: (2P, N, E) -- particle tokens followed by QFIM tokens
        seq_len, batch_size, _ = features.shape
        attn_mask = None
        if lorentz_vectors is not None and self.mod.pair_embed is not None:
            base_len = lorentz_vectors.size(-1)
            attn_mask = self.mod.pair_embed(lorentz_vectors).view(-1, base_len, base_len)
            if seq_len > base_len:
                pad = seq_len - base_len
                attn_mask = F.pad(attn_mask, (0, pad, 0, pad))
        # All 2P tokens are attended to. The trained model passes an all-False bool mask, which MultiheadAttention
        # converts to this all-zero float mask; passing it directly gives identical numbers without the
        # "mismatched key_padding_mask and attn_mask" deprecation warning.
        padding_mask = torch.zeros((batch_size, seq_len), dtype=features.dtype, device=features.device)

        with torch.amp.autocast('cuda', enabled=self.mod.use_amp):
            x = features
            for block in self.mod.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            cls_tokens = self.mod.cls_token.expand(1, x.size(1), -1)
            for block in self.mod.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.mod.norm(cls_tokens).squeeze(0)
            if self.mod.fc is None:
                return x_cls
            output = self.mod.fc(x_cls)
            if self.mod.for_inference:
                output = torch.softmax(output, dim=1)
            return output

    def forward(self, points, features, lorentz_vectors, mask, qfim):
        # trim particle features and QFIM together so both streams keep the same particles
        raw_features = torch.cat([features, qfim], dim=1)  # (N, C_pf + 90, P)
        raw_features, lorentz_vectors, mask, _ = self.trimmer(raw_features, lorentz_vectors, mask, None)
        features = raw_features[:, :self.pf_input_dim, :]
        qfim = raw_features[:, self.pf_input_dim:, :]

        features = self.pf_embed(features)                   # (P, N, E)
        qfim = self.qfim_token_alpha * self.qfim_embed(qfim)  # (P, N, E)
        features = torch.cat([features, qfim], dim=0)        # (2P, N, E)
        return self._forward_particle_concat(features, lorentz_vectors)


def _part_cfg(**kwargs):
    # get_model() defaults of networks/example_ParticleTransformer.py (identical in the qfim_simple file)
    cfg = dict(
        num_classes=2,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(kwargs)
    return cfg


def build_part(input_dim, **kwargs):
    """ParT as built by get_model() in networks/example_ParticleTransformer.py."""
    return ParTWrapper(**_part_cfg(input_dim=input_dim, **kwargs))


def build_quiver(pf_input_dim, qfim_input_dim=QFIM_INPUT_DIM, **kwargs):
    """QUIVER as built by get_model() in example_ParticleTransformer_qfim_simple.py with
    --network-option qfim_concat_mode 'post_embed_particle_concat'."""
    return QuiverWrapper(**_part_cfg(pf_input_dim=pf_input_dim, qfim_input_dim=qfim_input_dim, **kwargs))


def build_model(model, features):
    """model: 'ParT' or 'QUIVER'; features: 'kin' (3 particle features) or 'full' (17)."""
    features = features.lower()
    if model == 'ParT':
        return build_part(PF_INPUT_DIM[features])
    if model == 'QUIVER':
        return build_quiver(PF_INPUT_DIM[features])
    raise ValueError(f'unknown model {model!r}')


def load_model(model, features, weights, device='cpu'):
    """Build the network and load a thesis state dict (strict=True); returns the model in eval mode."""
    net = build_model(model, features)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def predict(model, data, features, batch_size=512, device='cpu'):
    """Signal score softmax(logits)[:, 1] for every jet in ``data`` (dict of numpy arrays from the demo npz).

    Batches are formed in file order with weaver's test batch size (512), so the trimmer sees the same batches
    as ``weaver --predict`` on the same jets.
    """
    model.eval()
    quiver = isinstance(model, QuiverWrapper)
    keys = ['pf_points', f'pf_features_{features.lower()}', 'pf_vectors', 'pf_mask'] + (['qfim'] if quiver else [])
    n = len(data[keys[0]])
    scores = []
    for start in range(0, n, batch_size):
        inputs = [torch.from_numpy(np.ascontiguousarray(data[k][start:start + batch_size])).to(device) for k in keys]
        logits = model(*inputs)
        scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)
