import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from rsl_rl.modules.actor_critic import ActorCritic, get_activation

class ActorCriticTransformer(ActorCritic):
    def __init__(
        self,
        num_actor_obs,  # input size of actor transformer
        num_critic_obs,  # input size of critic transformer
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        dropout=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        num_actor_out = 256 #temporarily hardcoded
        num_critic_out = 256 #temporarily hardcoded
        
        super().__init__(
            num_actor_obs=num_actor_out,
            num_critic_obs=num_critic_out,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        self.is_transformer = True
        
        self.transformer_a = Transformer(num_actor_obs, num_actor_out, num_heads, num_layers, hidden_dim, dropout)
        self.transformer_c = Transformer(num_critic_obs, num_critic_out, num_heads, num_layers, hidden_dim, dropout)
        
        self.sliding_window_a = SlidingWindowBuffer(24, num_actor_obs)
        self.sliding_window_c = SlidingWindowBuffer(24, num_critic_obs)
        
        print(f"Actor Transformer: {self.transformer_a}")
        print(f"Critic Transformer: {self.transformer_c}")

    def act(self, observations, **kwargs):
        input_a = self.transformer_a(observations)
        return super().act(input_a)

    def act_inference(self, observations):
        input_a = self.transformer_a(observations)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, **kwargs):
        input_c = self.transformer_c(critic_observations)
        return super().evaluate(input_c)
    
    def reset(self, dones=None):
        self.transformer_a.reset(dones)
        self.transformer_c.reset(dones)

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_emb = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # print(f"x: {x.shape}")
        x = self.input_emb(x)  # [batch_size, input_size] -> [batch_size, hidden_dim]
        # print(f"x after input_emb: {x.shape}")
        x = self.positional_encoding(x)
        # print(f"x after positional_encoding: {x.shape}")
        x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, hidden_dim]
        # print(f"x after unsqueeze: {x.shape}")
        x = self.transformer_encoder(x)  # Transformer expects [seq_len, batch_size, feature_size]
        # print(f"x after transformer_encoder: {x.shape}")
        x = x.squeeze(1)  # Remove sequence dimension [batch_size, hidden_dim]
        # print(f"x after squeeze: {x.shape}")
        return self.output(x)
    
    def reset(self, dones=None):
        pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SlidingWindowBuffer:
    def __init__(self, window_size, feature_dim):
        self.window_size = window_size
        self.feature_dim = feature_dim
        # Initialize buffer with zeros
        self.buffer = torch.zeros((window_size, feature_dim))
        self.current_size = 0  # Track current size of actual data in the buffer

    def update(self, new_observation):
        # Update the buffer and the current size
        self.buffer = torch.roll(self.buffer, -1, 0)
        self.buffer[-1] = new_observation
        if self.current_size < self.window_size:
            self.current_size += 1

    def get_window(self):
        return self.buffer[-self.current_size:]  # Return only the filled part

    def get_mask(self):
        # Create a mask where only the entries corresponding to actual data are True
        mask = torch.zeros(self.window_size, dtype=torch.bool)
        mask[-self.current_size:] = True
        return mask