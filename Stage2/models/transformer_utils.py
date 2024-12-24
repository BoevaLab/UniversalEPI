import copy
import math
import warnings
from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF, Tensor

# This script is a modified version of the original script from the PyTorch v1.11.0 repository 
# (https://pytorch.org/docs/1.11/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)

############################################################
##### Basis Functions for Relative Positional Encoding #####
############################################################


def _prepend_dims(x, num_dims):
    """Adds num_dims dimensions to x
    For example: if x is of shape (2,3) and num_dims is 2,
        output dimension will be (1,1,2,3).
    """
    return torch.reshape(x, shape=tuple([1] * num_dims + list(x.shape)))


def positional_features_exponential(
    positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None, min_half_life: Optional[float] = 3.0
):
    """Create exponentially decaying positional weights.
    Args:
        positions: Position tensor (arbitrary shape).
        feature_size: Number of basis functions to use.
        seq_length: Sequence length.
        min_half_life: Smallest exponential half life in the grid of half lives.
    Returns:
        A Tensor with shape [2 * seq_length - 1, feature_size].
    """
    if seq_length is None:
        seq_length = torch.max(torch.abs(positions)) + 1
    if type(seq_length) is int or type(seq_length) is float:
        seq_length = torch.as_tensor(seq_length)
    # Grid of half lifes from [3, seq_length / 2] with feature_size
    # distributed on the log scale.
    seq_length = seq_length.type(torch.float32)
    max_range = torch.log(seq_length) / torch.log(torch.as_tensor(2.0))
    half_life = torch.pow(2.0, torch.linspace(min_half_life, max_range.item(), feature_size))
    half_life = _prepend_dims(half_life, len(positions.squeeze().shape))
    positions = torch.abs(positions)
    outputs = torch.exp(-torch.log(torch.as_tensor(2.0)) / half_life * positions.unsqueeze(-1))
    # assert outputs.shape == (torch.ones(int(2*seq_length.item()-1), feature_size)).shape
    return outputs


def positional_features_central_mask(positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None):
    """Positional features using a central mask (allow only central features)."""
    center_widths = torch.pow(2.0, torch.arange(1 + 11, feature_size + 1 + 11))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, len(positions.squeeze().shape))
    outputs = center_widths > (torch.abs(positions).unsqueeze(-1)).type(torch.float32)
    # assert outputs.shape == (torch.ones(2*seq_length-1, feature_size)).shape
    return outputs


def _xlogy(x, y):
    z = torch.zeros(())
    if x.device.type == "cuda":
        z.to(x.get_device())
    return x * torch.where(x == 0.0, z, torch.log(y))


def _gamma_pdf(x, concentration, rate):
    """Gamma probability distribution function: p(x|concentration, rate)."""
    log_unnormalized_prob = _xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(
    positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None, stddev=None, start_mean=None
):
    """Positional features computed using the gamma distributions."""
    if seq_length is None:
        seq_length = torch.max(torch.abs(positions)) + 1
    if stddev is None:
        stddev = seq_length / (2 * feature_size)
    if start_mean is None:
        start_mean = seq_length / feature_size
    mean = torch.linspace(start_mean, seq_length, feature_size)
    mean = _prepend_dims(mean, len(positions.squeeze().shape))
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = _gamma_pdf(torch.abs(positions.type(torch.float32)).unsqueeze(-1), concentration, rate)
    probabilities += 1e-8  # To ensure numerical stability.
    outputs = probabilities / torch.max(probabilities, axis=1, keepdims=True)[0]
    return outputs


def positional_features_cosine(
    positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None, bin_size: Optional[int] = None
):
    """Cosine positional features."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    periodicity = 1.25 * torch.pow(2.0, torch.range(0, feature_size))
    periodicity = periodicity.type(torch.float32)
    periodicity = _prepend_dims(periodicity, len(positions.squeeze().shape))

    outputs = torch.cos(2 * np.pi * positions.unsqueeze(-1) / periodicity)
    assert outputs.shape == (torch.ones(2 * seq_length - 1, feature_size)).shape
    return outputs


def positional_features_linear_masks(positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None):
    """Exponentially increasing point focuses."""
    distances = torch.range(0, feature_size).type(torch.float32)
    distances = _prepend_dims(distances, len(positions.squeeze().shape))
    outputs = distances == torch.abs(positions.unsqueeze(-1)).type(torch.float32)

    assert outputs.shape == (torch.ones(2 * seq_length - 1, feature_size)).shape
    return outputs


def positional_features_sin_cos(
    positions: torch.Tensor, feature_size: int, seq_length: Optional[int] = None, max_time=10000.0
):
    """Sine/cosine positional encodings."""
    if feature_size % 2 != 0:
        raise ValueError("feature_size needs to be divisible by 2.")
    i = torch.arange(0, feature_size, 2).type(torch.float32)
    i = _prepend_dims(i, len(positions.squeeze().shape))

    # Concat sines and cosines and return.
    outputs = torch.concat(
        [
            torch.sin(positions.unsqueeze(-1) / max_time ** (i / feature_size)),
            torch.cos(positions.unsqueeze(-1) / max_time ** (i / feature_size)),
        ],
        -1,
    )

    return outputs


def _get_positional_feature_function(name):
    """Returns positional feature functions."""
    available = {
        "positional_features_exponential": positional_features_exponential,
        "positional_features_central_mask": positional_features_central_mask,
        "positional_features_gamma": positional_features_gamma,
        "positional_features_cosine": positional_features_cosine,
        "positional_features_linear_masks": positional_features_linear_masks,
        "positional_features_sin_cos": positional_features_sin_cos,
    }
    if name not in available:
        raise ValueError(f"Function {name} not available in {available.keys()}")
    return available[name]


def _positional_features_all(positions, feature_size, seq_length, feature_functions):
    r"""
    Compute relative positional encodings/features.
    Each positional feature function will compute/provide the same fraction of
    features, making up the total of feature_size.

    Args:
        positions: Tensor of relative positions of arbitrary shape.
        feature_size: Total number of basis functions.
        seq_length: Sequence length denoting the characteristic length that
            the individual positional features can use. This is required since the
            parametrization of the input features should be independent of `positions`
            while it could still require to use the total number of features.
        feature_functions: List of different feature functions to use. Each function
            will take as argument: positions, sequence length and number of features
            to compute.

    Returns:
        Tensor of shape: `positions.shape + (feature_size,)`.
    """
    num_components = 2 * len(feature_functions)  # 2 per each basis function (symmetric & non-symmetric)

    if feature_size % num_components != 0:
        raise ValueError(f"feature_size has to be divisible by {num_components}")
    feature_functions = [_get_positional_feature_function(f) for f in feature_functions]
    num_basis_per_class = feature_size // num_components
    embeddings = torch.concat(
        [f(torch.abs(positions), num_basis_per_class, seq_length) for f in feature_functions], axis=-1
    )
    embeddings = torch.concat([embeddings, torch.sign(positions).unsqueeze(-1) * embeddings], axis=-1)
    return embeddings


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Shape:
        - Input: :math:`(*, in\_features)` where `*` means any number of
          additional dimensions, including none
        - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
        - Bias: :math:`(out\_features)` or :math:`()`
        - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
    """
    return torch._C._nn.linear(input, weight, bias)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    See :class:`~torch.nn.Dropout` for details.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax function.
    Softmax is defined as:
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.
    See :class:`~torch.nn.Softmax` for more details.
    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


def _get_distances(meta, i, j, resolution, max_size):
    tmp = (meta[i, :] - meta[i, j]) / resolution
    tmp = tmp.to(int)
    tmp = torch.clamp(tmp, -max_size, max_size)
    tmp = torch.add(tmp, max_size)
    return tmp.tolist()


def _get_indices(meta, max_dist, resolution):
    meta = meta[:, :, 1:]  # [batch, seq_len, start:end]
    meta = meta.mean(2)  # [batch, seq_len]

    meta = meta.type(torch.IntTensor)
    batch_size, seq_len = meta.shape

    max_size = max_dist / resolution

    out_list = [[_get_distances(meta, i, j, resolution, max_size) for j in range(seq_len)] for i in range(batch_size)]
    return out_list


def _filter_r_k(meta, r_k, max_dist, resolution):
    max_size = max_dist / resolution

    meta = meta[:, :, 1:]  # [batch, seq_len, start:end]
    meta = meta.mean(2)  # [batch, seq_len]
    meta = torch.div(meta, resolution)
    meta = meta.type(torch.IntTensor)
    batch_size, seq_len = meta.shape
    heads, _, head_dim = r_k.shape

    meta_expanded = meta.unsqueeze(2).repeat(1, 1, seq_len).transpose(2, 1)
    distance_mat = meta_expanded - meta[:, :, None]
    distance_mat = distance_mat.to(int)
    distance_mat = torch.clamp(distance_mat, -max_size + 1, max_size - 1)
    ind_mat = torch.add(distance_mat, (max_size - 1))  # [batch, seq_len, seq_len]
    ind_mat = ind_mat.flatten().long()

    r_k = r_k[:, ind_mat, :].view(batch_size, seq_len, heads, seq_len, head_dim)

    return r_k


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def _rel_shift(x):
    bh, l, _, device, dtype = *x.shape, x.device, x.dtype
    kwargs = {"device": device, "dtype": dtype}
    col_pad = torch.zeros((bh, l, 1), **kwargs)
    x = torch.cat((x, col_pad), dim=2)
    flat_x = einops.rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((bh, l - 1), **kwargs)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.view(bh, l + 1, 2 * l - 1)
    final_x = final_x[:, :l, (l - 1) :]
    return final_x


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    weight_rk: Optional[Tensor],
    weight_rv: Optional[Tensor],
    bias_rw: Optional[Tensor],
    bias_rr: Optional[Tensor],
    weight_key: Optional[Tensor],
    weight_rel: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    relative_positions=False,
    relative_position_functions=None,
    num_relative_position_features=0,
    positional_dropout=0.0,
    meta: Tensor = None,
    max_dist=1000000,
    resolution=5000,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0
        positional_dropout = 0.0
    # get positional encoding

    if meta is not None:
        use_distance = True
    else:
        use_distance = False

    if relative_positions:
        if use_distance:
            max_size = max_dist / resolution

            meta = meta[:, :, 1:]  # [batch, seq_len, start:end]
            meta = meta.mean(2)  # [batch, seq_len]
            meta = torch.div(meta, resolution)
            meta = meta.type(torch.IntTensor)

            meta_expanded = meta.unsqueeze(2).repeat(1, 1, src_len).transpose(2, 1)
            distance_mat = meta_expanded - meta[:, :, None]
            distance_mat = distance_mat.to(int)
            distance_mat = torch.clamp(distance_mat, -max_size + 1, max_size - 1)
            positional_encodings = _positional_features_all(
                positions=distance_mat.flatten(),
                feature_size=num_relative_position_features,
                seq_length=src_len,
                feature_functions=relative_position_functions,
            )

            # r_k = r_k[:,ind_mat,:].view(batch_size, seq_len, heads, seq_len, head_dim)
            positional_encodings = dropout(
                positional_encodings, p=positional_dropout
            )  # [len.distance, num_relative_position_features]
            positional_encodings = torch.stack([positional_encodings] * num_heads)
            positional_encodings = positional_encodings.view(
                *positional_encodings.shape[:1], -1, *positional_encodings.shape[-1:]
            ).type(
                torch.float32
            )  # [num_heads, len.distance, num_relative_position_features]
            r_k = linear(positional_encodings.cuda(), weight_rk)
            r_k = r_k.view(bsz, src_len, num_heads, src_len, head_dim)
            r_k = torch.mean(r_k, 3)
            r_k = r_k.view(bsz * num_heads, head_dim, -1)

        else:
            distances = torch.arange(-src_len + 1, src_len)
            positional_encodings = _positional_features_all(
                positions=distances,
                feature_size=num_relative_position_features,
                seq_length=src_len,
                feature_functions=relative_position_functions,
            )
            positional_encodings = dropout(
                positional_encodings, p=positional_dropout
            )  # [len.distance, num_relative_position_features]
            positional_encodings = torch.stack([positional_encodings] * num_heads)
            positional_encodings = positional_encodings.view(
                *positional_encodings.shape[:1], -1, *positional_encodings.shape[-1:]
            )  # [num_heads, len.distance, num_relative_position_features]
            r_k = linear(positional_encodings, weight_rk)

        q = q / math.sqrt(q.shape[-1])
        content_attn = torch.bmm(q + bias_rw, k.transpose(-2, -1))
        # content_attn  = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            content_attn += attn_mask
        emb = q + bias_rr
        # emb = q
        if use_distance:
            relative_attn = torch.bmm(emb, r_k) + linear(k, weight_key) + linear(r_k.transpose(-2, -1), weight_rel)
        else:
            relative_attn = torch.einsum("x t d, h r d -> x t r", emb, r_k)
            relative_attn = _rel_shift(relative_attn)
        attn = content_attn + relative_attn
        attn = softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p)
        attn_output = torch.bmm(attn, v)
        attn_output_weights = attn
    else:
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    # calculate out projection
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        relative_positions=False,
        meta_flg=False,
        relative_position_functions=None,
        positional_dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        seq_len=None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.relative_positions = relative_positions
        self.relative_position_functions = relative_position_functions
        self.num_relative_position_features = 0
        # num_relative_position_features need to be divisible by number of relative positional functions * 2 (symmetric and asymmetric)
        self.weight_rk, self.weight_rv, self.bias_rw, self.bias_rr, self.weight_key, self.weight_rel = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.max_dist = 2000000
        self.resolution = 5000

        if self.relative_positions:
            if self.relative_position_functions is None:
                if meta_flg:
                    self.relative_position_functions = [
                        # 'positional_features_exponential',
                        # 'positional_features_gamma',
                        "positional_features_central_mask"
                    ]
                else:
                    self.relative_position_functions = ["positional_features_sin_cos"]
            if self.seq_len is None:
                raise ValueError('Sequence length cannot be "None" while using relative positional encoding.')
            divisibleby = 2 * len(self.relative_position_functions)
            self.num_relative_position_features = int((self.embed_dim // divisibleby) * divisibleby)
            self.weight_rk = nn.Parameter(
                torch.empty((self.head_dim, self.num_relative_position_features), **factory_kwargs)
            )
            self.weight_rv = nn.Parameter(torch.empty((self.seq_len, self.seq_len * self.seq_len), **factory_kwargs))
            self.bias_rw = nn.Parameter(torch.empty((1, self.seq_len, self.head_dim), **factory_kwargs))  #
            self.bias_rr = nn.Parameter(torch.empty((1, self.seq_len, self.head_dim), **factory_kwargs))  #
            # self.bias_rw = nn.Parameter(torch.empty((512*self.num_heads, self.seq_len, self.head_dim), **factory_kwargs)) #
            # self.bias_rr = nn.Parameter(torch.empty((512*self.num_heads, self.seq_len, self.head_dim), **factory_kwargs)) #
            self.weight_key = nn.Parameter(torch.empty((self.seq_len, self.head_dim), **factory_kwargs))
            self.weight_rel = nn.Parameter(torch.empty((self.seq_len, self.head_dim), **factory_kwargs))
        self.positional_dropout = positional_dropout

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.relative_positions:
            nn.init.constant_(self.bias_rw, 0.0)
            nn.init.constant_(self.bias_rr, 0.0)
            nn.init.xavier_uniform_(self.weight_rk)
            nn.init.xavier_uniform_(self.weight_rv)
            nn.init.xavier_uniform_(self.weight_key)
            nn.init.xavier_uniform_(self.weight_rel)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        meta: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
                when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
                and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
                key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
                :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
                :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
              :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
              the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
            - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
              size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
              when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # print(self.training)

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.weight_rk,
                self.weight_rv,
                self.bias_rw,
                self.bias_rr,
                self.weight_key,
                self.weight_rel,
                training=self.training,
                relative_positions=self.relative_positions,
                relative_position_functions=self.relative_position_functions,
                num_relative_position_features=self.num_relative_position_features,
                positional_dropout=self.positional_dropout,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                meta=meta,
                max_dist=self.max_dist,
                resolution=self.resolution,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.weight_rk,
                self.weight_rv,
                self.bias_rw,
                self.bias_rr,
                self.weight_key,
                self.weight_rel,
                training=self.training,
                relative_positions=self.relative_positions,
                relative_position_functions=self.relative_position_functions,
                num_relative_position_features=self.num_relative_position_features,
                positional_dropout=self.positional_dropout,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                meta=meta,
                max_dist=self.max_dist,
                resolution=self.resolution,
            )
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        relative_positions=False,
        meta_flg=False,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
        get_attn=False,
        seq_len=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            relative_positions=relative_positions,
            meta_flg=meta_flg,
            seq_len=seq_len,
            **factory_kwargs,
        )
        self.get_attn = get_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        meta: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            meta: genomic coordinates for relative positional encoding (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x_a, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, meta)
            x = x + x_a
            x = x + self._ff_block(self.norm2(x))
        else:
            x_a, attn = self._sa_block(x, src_mask, src_key_padding_mask, meta)
            x = self.norm1(x + x_a)
            x = self.norm2(x + self._ff_block(x))

        return x, attn

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], meta: Optional[Tensor] = None
    ) -> Tensor:
        attn_wt = None
        if self.get_attn:
            x, attn_wt = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, meta=meta)
        else:
            x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, meta=meta)
        return self.dropout1(x), attn_wt

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        meta: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            meta: genomic coordinates for relative positional encoding (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attn_wts = dict()

        for i, mod in enumerate(self.layers):
            output, attn_wt = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, meta=meta)

            if attn_wt is not None:
                attn_wts[f"Layer-{i}"] = attn_wt

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_wts
