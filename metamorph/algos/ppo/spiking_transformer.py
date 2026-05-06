import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("Unsupported activation: {}".format(activation))


def _build_spike_neuron(neuron_type, tau, detach_reset, backend):
    try:
        if neuron_type == "lif":
            return MultiStepLIFNode(
                tau=tau, detach_reset=detach_reset, backend=backend
            )
        elif neuron_type == "plif":
            return MultiStepParametricLIFNode(
                init_tau=tau, detach_reset=detach_reset, backend=backend
            )
    except Exception as exc:
        if backend != "torch":
            print(
                "[SpikingTransformer] Failed to initialize backend '{}': {}. "
                "Falling back to 'torch'.".format(backend, exc),
                flush=True,
            )
            return _build_spike_neuron(neuron_type, tau, detach_reset, "torch")
    raise ValueError("Unsupported spike neuron type: {}".format(neuron_type))


class SpikingTransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SpikingTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for l in self.layers:
            output = l(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None):
        attention_maps = []
        output = src

        for l in self.layers:
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True,
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps


class SpikingTransformerEncoderLayerResidual(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        spike_t=4,
        spike_neuron="lif",
        spike_tau=2.0,
        detach_reset=True,
        backend="cupy",
    ):
        super(SpikingTransformerEncoderLayerResidual, self).__init__()
        if d_model % nhead != 0:
            raise ValueError(
                "d_model ({}) must be divisible by nhead ({})".format(
                    d_model, nhead
                )
            )

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.spike_t = spike_t
        self.spike_neuron = spike_neuron
        self.spike_tau = spike_tau
        self.detach_reset = detach_reset
        self.backend = backend

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_lif = _build_spike_neuron(
            spike_neuron, spike_tau, detach_reset, backend
        )
        self.k_lif = _build_spike_neuron(
            spike_neuron, spike_tau, detach_reset, backend
        )
        self.v_lif = _build_spike_neuron(
            spike_neuron, spike_tau, detach_reset, backend
        )
        self.attn_out_lif = _build_spike_neuron(
            spike_neuron, spike_tau, detach_reset, backend
        )
        self.ffn_lif = _build_spike_neuron(
            spike_neuron, spike_tau, detach_reset, backend
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(SpikingTransformerEncoderLayerResidual, self).__setstate__(state)

    def _repeat_in_time(self, src):
        return src.unsqueeze(0).repeat(self.spike_t, 1, 1, 1)

    def _set_backend(self, backend):
        self.backend = backend
        for node_name in ["q_lif", "k_lif", "v_lif", "attn_out_lif", "ffn_lif"]:
            getattr(self, node_name).backend = backend

    def _fallback_to_torch(self, exc):
        if self.backend == "torch":
            raise exc
        print(
            "[SpikingTransformer] Backend '{}' failed during forward: {}. "
            "Switching to 'torch' and continuing.".format(self.backend, exc),
            flush=True,
        )
        self._set_backend("torch")
        functional.reset_net(self)

    def _reshape_qkv(self, x):
        T, L, B, _ = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .reshape(T, B, L, self.nhead, self.head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

    def _apply_attention_mask(self, attn_logits, src_mask, src_key_padding_mask):
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(
                    src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    torch.finfo(attn_logits.dtype).min,
                )
            else:
                attn_logits = attn_logits + src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if src_key_padding_mask is not None:
            padding_mask = (
                src_key_padding_mask.to(torch.bool)
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(2)
            )
            attn_logits = attn_logits.masked_fill(
                padding_mask, torch.finfo(attn_logits.dtype).min
            )

        return attn_logits

    def _apply_key_padding_mask(self, x, src_key_padding_mask):
        if src_key_padding_mask is None:
            return x
        keep_mask = (
            (~src_key_padding_mask.to(torch.bool))
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(-1)
            .to(x.dtype)
        )
        return x * keep_mask

    def _aggregate_kv(self, kv, src_mask):
        if src_mask is None:
            return kv.sum(dim=-2, keepdim=True)

        if src_mask.dtype == torch.bool:
            weights = (~src_mask).to(kv.dtype)
        else:
            weights = torch.isfinite(src_mask).to(kv.dtype)

        return torch.einsum("qk,tbhkd->tbhqd", weights, kv)

    def _get_attention_map(self, q, k, src_mask, src_key_padding_mask):
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = self._apply_attention_mask(
            attn_logits, src_mask, src_key_padding_mask
        )
        attn_weights = torch.softmax(attn_logits, dim=-1)
        return attn_weights.mean(dim=(0, 2))

    def _forward_impl(
        self, src, src_mask=None, src_key_padding_mask=None, return_attention=False
    ):
        functional.reset_net(self)

        src2 = self.norm1(src)
        spike_src = self._repeat_in_time(src2)
        T, L, B, _ = spike_src.shape
        spike_src_flat = spike_src.reshape(T, L * B, self.d_model)

        q = self.q_proj(spike_src_flat)
        q = self.q_lif(q).reshape(T, L, B, self.d_model)
        q = self._reshape_qkv(q)

        k = self.k_proj(spike_src_flat)
        k = self.k_lif(k).reshape(T, L, B, self.d_model)
        k = self._reshape_qkv(k)

        v = self.v_proj(spike_src_flat)
        v = self.v_lif(v).reshape(T, L, B, self.d_model)
        v = self._reshape_qkv(v)

        q = self._apply_key_padding_mask(q, src_key_padding_mask)
        k = self._apply_key_padding_mask(k, src_key_padding_mask)
        v = self._apply_key_padding_mask(v, src_key_padding_mask)

        attention_map = None
        if return_attention:
            attention_map = self._get_attention_map(
                q, k, src_mask, src_key_padding_mask
            )

        kv = k.mul(v)
        kv = self._aggregate_kv(kv, src_mask)
        kv = self.attn_dropout(kv)
        attn_output = q.mul(kv)
        attn_output = (
            attn_output.permute(0, 3, 1, 2, 4)
            .reshape(T, L * B, self.d_model)
            .contiguous()
        )
        attn_output = self.attn_out_lif(attn_output)
        attn_output = self.out_proj(attn_output).reshape(T, L, B, self.d_model)
        src = src + self.dropout1(attn_output.mean(0))

        src2 = self.norm2(src)
        spike_src2 = self._repeat_in_time(src2).reshape(T, L * B, self.d_model)
        src2 = self.linear1(spike_src2)
        src2 = self.activation(src2)
        src2 = self.ffn_lif(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2).reshape(T, L, B, self.d_model)
        src = src + self.dropout2(src2.mean(0))

        if return_attention:
            return src, attention_map
        else:
            return src

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, return_attention=False
    ):
        try:
            return self._forward_impl(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
            )
        except Exception as exc:
            self._fallback_to_torch(exc)
            return self._forward_impl(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
            )
