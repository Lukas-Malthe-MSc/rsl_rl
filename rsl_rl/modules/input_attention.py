import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, masks=None):
        # Detect shape based on input dimensions
        if x.dim() == 3:
            time_steps, batch_size, features = x.shape
        elif x.dim() == 2:
            batch_size, features = x.shape
            time_steps = 1  # Assume single timestep for inference mode
            x = x.unsqueeze(0)  # Add a time dimension for uniform processing
        else:
            raise ValueError("Unsupported tensor shape")

        queries = self.query_projection(x).view(time_steps, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_projection(x).view(time_steps, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_projection(x).view(time_steps, batch_size, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(time_steps, batch_size, -1)

        if masks is not None and masks.dim() > 1:
            # Apply masks to filter out padding in batch mode
            masks = masks.unsqueeze(1).expand(-1, self.num_heads, -1)
            context = context * masks

        # Squeeze the time dimension if it was originally a 2D input
        if x.dim() == 2:
            context = context.squeeze(0)

        context = torch.squeeze(context)
        return context
