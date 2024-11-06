import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """

    BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM = queries.shape

    raw_scores = torch.matmul(queries, keys.transpose(1, 2)) / (HIDDEN_DIM**(1/2))
    
    attention_scores = F.softmax(raw_scores, dim=-1)
    
    attention_output = torch.matmul(attention_scores, values)
    
    return attention_output


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """

    BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD = queries.shape

    attention_heads = []
    
    for head in range(N_HEADS):
        q_head = queries[:, head, :, :]  
        k_head = keys[:, head, :, :]     
        v_head = values[:, head, :, :] 

        attention_weights = torch.matmul(q_head, k_head.transpose(1, 2)) / (DIM_PER_HEAD ** 0.5)

        attention_scores = torch.softmax(attention_weights, dim=-1)

        attention_output = torch.matmul(attention_scores, v_head) 

        attention_heads.append(attention_output)

    attention_output = torch.cat(attention_heads, dim=-1) 

    output = torch.matmul(attention_output, projection_matrix.T) 
    
    return output


def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = x.shape

    theta = 10000 ** -(torch.arange(0, DIM_PER_HEAD, 2).float() / DIM_PER_HEAD)

    m = torch.arange(SEQ_LENGTH)

    angles = m.unsqueeze(1) * theta.unsqueeze(0)

    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2) 
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2)

    x_reshaped = x.reshape(BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD // 2, 2)

    x_cos = x_reshaped[..., 0] * cos_vals - x_reshaped[..., 1] * sin_vals
    x_sin = x_reshaped[..., 1] * cos_vals + x_reshaped[..., 0] * sin_vals

    x_rotared = torch.stack((x_cos, x_sin), dim=-1).flatten(3)
    
    return x_rotared
