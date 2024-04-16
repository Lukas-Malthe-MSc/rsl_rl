import torch
import torch.nn as nn
import torch.nn.functional as F
# import math

class InputAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.num_heads = num_heads
        # self.hidden_dim = hidden_dim
        # self.depth = hidden_dim // num_heads

    def forward(self, x):
        # If input is 3D, flatten it to 2D, preserving the last dimension
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(-1, original_shape[2])  # Flatten time and batch together

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(keys.size(-1))
        attention_weights = F.softmax(scores, dim=-1)

        # Create the context vector
        context = torch.matmul(attention_weights, values)

        # Reshape context to match original input structure, except the last dimension becomes `hidden_dim`
        if len(original_shape) == 3:
            context = context.view(original_shape[0], original_shape[1], -1)
        else:
            context = context.view(original_shape[0], -1)

        return context


    # def forward(self, x):
    #     # Handling variable input dimensions
    #     original_shape = x.shape
    #     if len(original_shape) > 2:
    #         # Flatten all dimensions except the last (feature dimension)
    #         x = x.reshape(-1, original_shape[-1])  # Use reshape to handle non-contiguous tensors

    #     B, N = x.size()
    #     # Linear projections
    #     Q = self.query_projection(x)  # [B, hidden_dim]
    #     K = self.key_projection(x)    # [B, hidden_dim]
    #     V = self.value_projection(x)  # [B, hidden_dim]

    #     # Reshape and transpose for multi-head attention
    #     Q = Q.view(B, self.num_heads, self.depth).permute(1, 0, 2)  # [num_heads, B, depth]
    #     K = K.view(B, self.num_heads, self.depth).permute(1, 0, 2)  # [num_heads, B, depth]
    #     V = V.view(B, self.num_heads, self.depth).permute(1, 0, 2)  # [num_heads, B, depth]

    #     # Scaled dot-product attention
    #     scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.depth)  # [num_heads, B, B]
    #     attention_weights = F.softmax(scores, dim=-1)  # [num_heads, B, B]

    #     # Apply attention weights to values
    #     context = torch.bmm(attention_weights, V)  # [num_heads, B, depth]
    #     context = context.permute(1, 0, 2).contiguous().view(B, -1)  # Flatten back to [B, hidden_dim]
        
    #     print
        return context

    def get_attention_weights(self, x):
        # Linear projections
        Q = self.query_projection(x)
