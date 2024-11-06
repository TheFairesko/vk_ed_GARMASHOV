import torch
import math


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    
    _, seq_len, num_heads, hidden_dim = query.shape
    _, kv_seq_len, num_kv_heads, _ = key.shape

    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    query = query.permute(0, 2, 1, 3)

    if num_kv_heads > num_heads:
        raise ValueError("The number of KV heads cannot be greater than the number of query heads")

    assert num_heads % num_kv_heads == 0, "Number of heads must be divisible by number of key-value heads"
    group_size = num_heads // num_kv_heads

    attention_heads = []
    all_attention_weights = []
    
    for group in range(num_kv_heads):
        q_group = query[:, group * group_size : (group + 1) * group_size, :, :]
        k_group = key[:, group, :, :]
        v_group = value[:, group, :, :]

        attention_raw_scores = torch.matmul(q_group, k_group.transpose(1, 2)) / (hidden_dim ** 0.5)

        if is_causal:
            mask = torch.tril(torch.ones(seq_len, kv_seq_len), diagonal=0).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
            attention_raw_scores = attention_raw_scores.masked_fill_(~mask, float('-inf'))

        attention_scores = torch.softmax(attention_raw_scores, dim=-1)

        if need_weights:
            all_attention_weights.append(attention_scores)

        attention_output = torch.matmul(attention_scores, v_group) 

        attention_heads.append(attention_output)

    attention_output = torch.cat(attention_heads, dim=1).permute(0, 2, 1, 3)
    
    if need_weights:
        all_attention_weights = torch.cat(all_attention_weights, dim=1)
        return (attention_output, all_attention_weights)

    else:
        return (attention_output, None)