import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """

    addict_part = torch.zeros((seq_len, seq_len))

    for k in range(1, seq_len):
        addict_part += torch.diag(torch.full((seq_len - k,), -k), diagonal=-k)
        addict_part += torch.diag(torch.full((seq_len - k,), k), diagonal=k)

    m = torch.pow(2.0, -torch.linspace(8/num_heads, 8, num_heads).float())
    alibi = addict_part.unsqueeze(0) * m.view(num_heads, 1, 1)

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
