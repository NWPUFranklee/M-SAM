import torch
import torch.nn as nn
import torch.nn.functional as F
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

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将预计算的旋转位置编码应用到 Query 和 Key 向量上。

    Args:
        xq: Query 张量，形状通常为 [batch_size, seq_len, dim]。
        xk: Key 张量，形状通常为 [batch_size, seq_len, dim]。
        freqs_cis: 预计算的复数频率张量，形状为 [seq_len, dim // 2]。

    Returns:
        应用RoPE后的 Query 和 Key 张量。
    """
    # 确保 freqs_cis 的序列长度与 xq, xk 当前的序列长度匹配
    # 这是为了处理不同长度的输入序列，从预计算的最大长度中取需要的长度部分
    # freqs_cis_current = freqs_cis[:xq.shape[1], :]
    # 在这里，我们假设传入的 freqs_cis 就是当前 seq_len 对应的部分，
    # 或者调用方负责切片。根据 Attention 类的 forward 实现，调用方会进行切片。

    # 将 xq, xk 的最后一个维度 (dim) 重塑，分成 dim//2 组，每组2个元素
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 将每组两个元素的实数张量视为一个复数张量
    # xq_.shape = [batch_size, seq_len, dim // 2] (复数)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作：复数乘法 (xq_ * freqs_cis)
    # freqs_cis 的形状是 [seq_len, dim // 2]，与 xq_ 和 xk_ 的后两个维度匹配
    # 通过广播机制，freqs_cis 会被扩展到 batch_size 维度
    # 旋转后的结果仍然是复数张量
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    # 将旋转后的复数结果转换回实数表示
    # 形状变为 [batch_size, seq_len, dim // 2, 2]
    xq_out = torch.view_as_real(xq_out)
    xk_out = torch.view_as_real(xk_out)

    # 将倒数第二个和最后一个维度展平，恢复到原始的 dim 维度
    # 形状变为 [batch_size, seq_len, dim]
    xq_out = xq_out.flatten(2)
    xk_out = xk_out.flatten(2)
    
    # 确保输出张量的数据类型与输入张量一致
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    带有旋转位置编码 (RoPE) 的自注意力模块。
    """
    def __init__(self, dim: int, max_seq_len: int):
        """
        Args:
            dim: 词向量的维度。
            max_seq_len: 训练或使用的最大序列长度，用于预计算RoPE频率。
        """
        super().__init__()
        self.dim = dim
        
        # 补全 Linear(...) 占位符
        self.wq = nn.Linear(dim, dim, bias=False) # 通常 Q, K, V 投影没有偏置
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        
        # 预计算最大序列长度的 RoPE 频率
        # 注意：这里预计算的是 max_seq_len 长度的 freqs_cis
        self.register_buffer('freqs_cis', precompute_freqs_cis(dim, max_seq_len))

    def forward(self, x: torch.Tensor):
        """
        前向传播计算带 RoPE 的自注意力。

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]。

        Returns:
            自注意力输出张量，形状为 [batch_size, seq_len, dim]。
        """
        bsz, seqlen, current_dim = x.shape
        
        # 确保输入维度与初始化时的 dim 一致
        assert current_dim == self.dim, f"Input dimension {current_dim} does not match layer dimension {self.dim}"

        # 线性变换得到 Query, Key, Value
        # xq, xk, xv shape: [batch_size, seq_len, dim]
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 在应用注意力之前，对 xq 和 xk 应用旋转位置编码
        # 从预计算的 freqs_cis 中取出当前序列长度对应的部分
        freqs_cis_current = self.freqs_cis[:seqlen, :]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis_current)
        
        # 计算注意力分数 (Scaled Dot-Product)
        # scores.shape = (bsz, seqlen, seqlen)
        # 这里需要将 xk 转置，以便进行矩阵乘法 (batch_size, seq_len, dim) @ (batch_size, dim, seq_len)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.dim)
        
        # 应用 Softmax 获取注意力权重
        scores = F.softmax(scores.float(), dim=-1)
        
        # 将注意力权重应用于 Value 向量
        # output.shape = (bsz, seqlen, dim)
        output = torch.matmul(scores, xv)
        
        return output

# --- 测试函数 ---
def test_attention_with_rope():
    """
    测试带有 RoPE 的 Attention 类。
    """
    print("--- 开始测试 Attention 模块 ---")

    # 定义测试参数
    test_dim = 64         # 词向量维度 (需要是偶数)
    test_max_seq_len = 128  # 最大序列长度
    test_batch_size = 4     # 批次大小
    test_seq_len = 10      # 当前输入序列长度 (小于等于 test_max_seq_len)

    print(f"测试参数: dim={test_dim}, max_seq_len={test_max_seq_len}, batch_size={test_batch_size}, seq_len={test_seq_len}")

    # 创建 Attention 模块实例
    try:
        attention_module = Attention(dim=test_dim, max_seq_len=test_max_seq_len)
        print("Attention 模块创建成功。")
        # print(attention_module) # 可以打印模块结构
        # print(f"预计算的 freqs_cis 形状: {attention_module.freqs_cis.shape}")

    except Exception as e:
        print(f"创建 Attention 模块时发生错误: {e}")
        return

    # 创建一个模拟的输入张量
    # 形状为 [batch_size, seq_len, dim]
    try:
        input_tensor = torch.randn(test_batch_size, test_seq_len, test_dim)
        print(f"模拟输入张量形状: {input_tensor.shape}")
    except Exception as e:
        print(f"创建输入张量时发生错误: {e}")
        return

    # 将模块设置为评估模式 (虽然对于这个简单的测试影响不大)
    attention_module.eval()

    # 调用 forward 方法进行计算
    try:
        with torch.no_grad(): # 在测试时通常不需要计算梯度
            output_tensor = attention_module(input_tensor)

        print("前向传播计算成功。")
        print(f"输出张量形状: {output_tensor.shape}")

        # 验证输出形状是否正确
        expected_shape = (test_batch_size, test_seq_len, test_dim)
        if output_tensor.shape == expected_shape:
            print("输出形状与期望形状一致，测试通过。")
        else:
            print(f"输出形状 {output_tensor.shape} 与期望形状 {expected_shape} 不一致，测试失败。")

    except Exception as e:
        print(f"执行前向传播时发生错误: {e}")
        print("测试失败。")

    print("--- Attention 模块测试结束 ---")

# 在脚本直接运行时执行测试函数
if __name__ == "__main__":
    test_attention_with_rope()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Tuple

# # 确保 Tensor 类型可用
# Tensor = torch.Tensor

# # --- Rotary Positional Embedding (RoPE) Helper Functions ---

# # 沿用之前的 precompute_freqs_cis，但要注意这里的 dim 将是每个头的维度 c_per_head
# def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> Tensor:
#     """
#     预计算旋转位置编码所需的复数频率张量。
#     这里的 dim 是每个头的维度 (c_per_head)。
#     """
#     # 计算词向量元素两两分组之后，每组元素对应的旋转角度的基础频率
#     # freqs = [1/(theta^(0/dim)), 1/(theta^(2/dim)), ..., 1/(theta^((dim-2)/dim))]
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

#     # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
#     t = torch.arange(seq_len, device=freqs.device)

#     # 计算每个位置 t 对应的频率张量
#     # freqs[i, j] = t[i] * freqs[j]
#     # freqs.shape = [seq_len, dim // 2]
#     freqs = torch.outer(t, freqs).float()

#     # 将频率转换为单位圆上的复数，形式为 cos(angle) + i*sin(angle)
#     # angle 在这里就是 freqs 张量中的值
#     # freqs_cis.shape = [seq_len, dim // 2]
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis

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
    return x_out.type_as(x)

# # --- Attention Class with RoPE ---

# class Attention(nn.Module):
#     """
#     An attention layer that allows for downscaling the size of the embedding
#     after projection to queries, keys, and values, and incorporates Rotary
#     Positional Embedding (RoPE).
#     """

#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         downsample_rate: int = 1,
#         max_seq_len: int = 1024, # 添加 max_seq_len 参数用于 RoPE 预计算
#     ) -> None:
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.internal_dim = embedding_dim // downsample_rate
#         self.num_heads = num_heads
        
#         # 计算每个头的维度
#         self.c_per_head = self.internal_dim // num_heads
        
#         assert self.internal_dim % num_heads == 0, "num_heads must divide internal_dim."
#         # RoPE 应用于 c_per_head 维度，该维度必须是偶数
#         assert self.c_per_head % 2 == 0, "embedding_dim // downsample_rate // num_heads must be even for RoPE."

#         self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=False) # 投影到 internal_dim
#         self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=False)
#         self.v_proj = nn.Linear(embedding_dim, self.internal_dim, bias=False)
#         self.out_proj = nn.Linear(self.internal_dim, embedding_dim) # 投影回 embedding_dim

#         # 预计算最大序列长度的 RoPE 频率
#         # freqs_cis 的维度是 c_per_head // 2
#         self.register_buffer('freqs_cis', precompute_freqs_cis(self.c_per_head, max_seq_len))


#     def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
#         """
#         Reshape input from [B, N_tokens, internal_dim] to [B, N_heads, N_tokens, C_per_head].
#         """
#         b, n_tokens, c = x.shape # c 是 internal_dim
#         x = x.reshape(b, n_tokens, num_heads, c // num_heads) # c // num_heads 是 C_per_head
#         return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

#     def _recombine_heads(self, x: Tensor) -> Tensor:
#         """
#         Reshape input from [B, N_heads, N_tokens, C_per_head] back to [B, N_tokens, internal_dim].
#         """
#         b, n_heads, n_tokens, c_per_head = x.shape
#         x = x.transpose(1, 2)
#         return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x internal_dim

#     def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#         """
#         前向传播计算带 RoPE 的自注意力。

#         Args:
#             q: Query 输入张量，形状为 [batch_size, seq_len, embedding_dim]。
#             k: Key 输入张量，形状为 [batch_size, seq_len, embedding_dim]。
#             v: Value 输入张量，形状为 [batch_size, seq_len, embedding_dim]。

#         Returns:
#             自注意力输出张量，形状为 [batch_size, seq_len, embedding_dim]。
#         """
#         # Input projections
#         # q, k, v shape after projection: [batch_size, seq_len, internal_dim]
#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)
        
#         print(q.shape)
#         # Separate into heads
#         # q, k, v shape after separating heads: [batch_size, num_heads, seq_len, c_per_head]
#         q = self._separate_heads(q, self.num_heads)
#         k = self._separate_heads(k, self.num_heads)
#         v = self._separate_heads(v, self.num_heads)

#         # --- 应用旋转位置编码 (RoPE) ---
#         # freqs_cis 形状是 [max_seq_len, c_per_head // 2]
#         # 取出当前序列长度对应的频率部分
#         seqlen = q.shape[2]
#         # 确保当前序列长度不超过预计算的最大长度
#         assert seqlen <= self.freqs_cis.shape[0], f"Sequence length ({seqlen}) exceeds max_seq_len ({self.freqs_cis.shape[0]}) for RoPE."
#         freqs_cis_current = self.freqs_cis[:seqlen, :]

#         # 将 RoPE 应用到 Query 和 Key 向量上
#         # q, k shape after RoPE: [batch_size, num_heads, seq_len, c_per_head]
#         q = apply_rotary_emb(q, freqs_cis_current)
#         k = apply_rotary_emb(k, freqs_cis_current) # 修复typo：frecs_cis_current -> freqs_cis_current

#         # --- Attention 计算 ---
#         # _, num_heads, seqlen, c_per_head = q.shape # 已经有了 seqlen 和 c_per_head
        
#         # 计算注意力分数 (Scaled Dot-Product)
#         # q @ k.permute(0, 1, 3, 2) 形状: [B, N_heads, N_tokens, N_tokens]
#         # (batch_size, num_heads, seq_len, c_per_head) @ (batch_size, num_heads, c_per_head, seq_len)
#         attn = q @ k.permute(0, 1, 3, 2)

#         # 缩放分数
#         attn = attn / math.sqrt(self.c_per_head) # 注意：这里是除以每个头维度的平方根

#         # 应用 Softmax 获取注意力权重
#         attn = torch.softmax(attn, dim=-1) # 形状: [B, N_heads, N_tokens, N_tokens]

#         # 将注意力权重应用于 Value 向量
#         # out = attn @ v 形状: [B, N_heads, N_tokens, C_per_head]
#         # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, c_per_head)
#         out = attn @ v

#         # Recombine heads
#         # out shape after recombining: [batch_size, seq_len, internal_dim]
#         out = self._recombine_heads(out)

#         # Output projection
#         # out shape after output projection: [batch_size, seq_len, embedding_dim]
#         out = self.out_proj(out)

#         return out

# # --- 测试函数 (可以沿用或修改之前的测试函数来测试这个新的 Attention 类) ---
# def test_attention_with_rope_v2():
#     """
#     测试带有 RoPE 和 Downsampling 的 Attention 类 (v2)。
#     """
#     print("--- 开始测试 Attention 模块 (v2 with RoPE & Downsampling) ---")

#     # 定义测试参数
#     test_embedding_dim = 256  # 输入/输出维度
#     test_num_heads = 8       # 注意力头数量
#     test_downsample_rate = 1 # 下采样率，internal_dim = 128 // 2 = 64
#     test_max_seq_len = 256   # 最大序列长度 for RoPE
#     test_batch_size = 4      # 批次大小
#     test_seq_len = 27       # 当前输入序列长度 (小于等于 test_max_seq_len)

#     # 计算每个头的维度
#     test_internal_dim = test_embedding_dim // test_downsample_rate
#     test_c_per_head = test_internal_dim // test_num_heads

#     print(f"测试参数: embedding_dim={test_embedding_dim}, num_heads={test_num_heads}, downsample_rate={test_downsample_rate}")
#     print(f"内部维度 internal_dim={test_internal_dim}, 每个头的维度 c_per_head={test_c_per_head}")
#     print(f"RoPE 最大序列长度={test_max_seq_len}, batch_size={test_batch_size}, 当前 seq_len={test_seq_len}")

#     # 验证 RoPE 要求
#     assert test_c_per_head % 2 == 0, "c_per_head must be even for RoPE."
#     assert test_internal_dim % test_num_heads == 0, "internal_dim must be divisible by num_heads."
#     assert test_seq_len <= test_max_seq_len, "Current seq_len exceeds max_seq_len for RoPE."


#     # 创建 Attention 模块实例
#     try:
#         attention_module = Attention(
#             embedding_dim=test_embedding_dim,
#             num_heads=test_num_heads,
#             downsample_rate=test_downsample_rate,
#             max_seq_len=test_max_seq_len
#         )
#         print("Attention 模块 (v2) 创建成功。")
#         # print(f"预计算的 freqs_cis 形状: {attention_module.freqs_cis.shape}")

#     except Exception as e:
#         print(f"创建 Attention 模块 (v2) 时发生错误: {e}")
#         return

#     # 创建一个模拟的输入张量 (Q, K, V 使用相同的输入进行自注意力测试)
#     # 形状为 [batch_size, seq_len, embedding_dim]
#     try:
#         input_tensor = torch.randn(test_batch_size, test_seq_len, test_embedding_dim)
#         print(f"模拟输入张量形状: {input_tensor.shape}")
#     except Exception as e:
#         print(f"创建输入张量时发生错误: {e}")
#         return

#     # 将模块设置为评估模式
#     attention_module.eval()

#     # 调用 forward 方法进行计算
#     try:
#         with torch.no_grad(): # 在测试时通常不需要计算梯度
#             output_tensor = attention_module(input_tensor, input_tensor, input_tensor) # 自注意力，Q=K=V=input

#         print("前向传播计算成功。")
#         print(f"输出张量形状: {output_tensor.shape}")

#         # 验证输出形状是否正确
#         expected_shape = (test_batch_size, test_seq_len, test_embedding_dim)
#         if output_tensor.shape == expected_shape:
#             print("输出形状与期望形状一致，测试通过。")
#         else:
#             print(f"输出形状 {output_tensor.shape} 与期望形状 {expected_shape} 不一致，测试失败。")

#     except Exception as e:
#         print(f"执行前向传播时发生错误: {e}")
#         print("测试失败。")

#     print("--- Attention 模块 (v2) 测试结束 ---")

# # 在脚本直接运行时执行测试函数
# if __name__ == "__main__":
#     test_attention_with_rope_v2()