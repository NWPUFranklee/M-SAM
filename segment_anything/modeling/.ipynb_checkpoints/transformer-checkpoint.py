# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
    
import math
from typing import Tuple

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    预计算旋转位置编码所需的复数频率张量。

    Args:
        dim: 词向量的维度。必须是偶数。
        seq_len: 需要计算的最大序列长度。
        theta: RoPE算法中的一个常数。
    
    Returns:
        形状为 [seq_len, dim // 2] 的复数张量，包含每个位置和每对维度的旋转信息。
    """
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度的基础频率
    # freqs = [1/(theta^(0/dim)), 1/(theta^(2/dim)), ..., 1/(theta^((dim-2)/dim))]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    
    # 计算每个位置 t 对应的频率张量
    # freqs[i, j] = t[i] * freqs[j]
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    
    # 将频率转换为单位圆上的复数，形式为 cos(angle) + i*sin(angle)
    # angle 在这里就是 freqs 张量中的值
    # freqs_cis.shape = [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# 修改 apply_rotary_emb 函数，以处理多头维度
def apply_rotary_emb(
    x: Tensor, # 可以是 xq 或 xk，形状为 [B, N_heads, N_tokens, C_per_head]
    freqs_cis: Tensor, # 形状为 [N_tokens, C_per_head // 2]
) -> Tensor:
    """
    将预计算的旋转位置编码应用到输入向量上。
    输入向量形状为 [B, N_heads, N_tokens, C_per_head]。
    """
    # 确保 freqs_cis 的序列长度与输入张量 x 的序列长度匹配
    seqlen = x.shape[2]
    # freqs_cis_current = freqs_cis[:seqlen, :] # 调用方会在传入前切片

    # 将输入的最后一个维度 (C_per_head) 重塑，分成 C_per_head//2 组，每组2个元素
    # x_.shape = [B, N_heads, N_tokens, C_per_head // 2, 2]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)

    # 将每组两个元素的实数张量视为一个复数张量
    # x_.shape = [B, N_heads, N_tokens, C_per_head // 2] (复数)
    x_ = torch.view_as_complex(x_)

    # 应用旋转操作：复数乘法 (x_ * freqs_cis)
    # x_ 的形状是 [B, N_heads, N_tokens, C_per_head // 2]
    # freqs_cis 的形状是 [N_tokens, C_per_head // 2]
    # PyTorch 的广播机制会使其在 B 和 N_heads 维度上进行广播匹配
    # 旋转后的结果仍然是复数张量
    x_out = x_ * freqs_cis

    # 将旋转后的复数结果转换回实数表示
    # 形状变为 [B, N_heads, N_tokens, C_per_head // 2, 2]
    x_out = torch.view_as_real(x_out)

    # 将倒数第二个和最后一个维度展平，恢复到原始的 C_per_head 维度
    # 形状变为 [B, N_heads, N_tokens, C_per_head]
    x_out = x_out.flatten(3) # 从第3个维度开始展平 (0-indexed)

    # 确保输出张量的数据类型与输入张量一致
    return x_out.type_as(x).flatten(2)


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # max_seq_len = 4096
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        # self.register_buffer('freqs_cis', precompute_freqs_cis(self.internal_dim, max_seq_len))
        
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        # bsz, seqlen_q, current_dim = q.shape
        # bsz, seqlen_k, current_dim = k.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # freqs_cis_current_q = self.freqs_cis[:seqlen_q, :]
        # freqs_cis_current_k = self.freqs_cis[:seqlen_k, :]
        # q = apply_rotary_emb(q, freqs_cis=freqs_cis_current_q)
        # k = apply_rotary_emb(k, freqs_cis=freqs_cis_current_k)
        
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
