from __future__ import annotations

import torch
import torch.nn as nn
import math
import numpy as np

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rich import print

class ActorCriticSelfAttention(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        attention_size=512,
        embedding_size=128,
        init_noise_std=1.0,
        **kwargs,
    ):

        super().__init__(
            num_actor_obs=attention_size,
            num_critic_obs=attention_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.attention_a = SelfAttention(input_size=num_actor_obs, embedding_size=embedding_size, attention_size=attention_size)
        self.attention_c = SelfAttention(input_size=num_critic_obs, embedding_size=embedding_size, attention_size=attention_size)

        self.actor.insert(0, self.attention_a)
        self.critic.insert(0, self.attention_c)
        
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

    def reset(self, dones=None):
        pass

    def act(self, observations, **kwargs):
        return super().act(observations=observations)

    def act_inference(self, observations):
        return super().act_inference(observations=observations)

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(critic_observations=critic_observations)




class SelfAttention(nn.Module):
    def __init__(self, input_size, embedding_size=4, attention_size=128):
        super(SelfAttention, self).__init__()
        self.in_features = input_size
        self.embedding = nn.Linear(1, embedding_size)
        self.attention_size = attention_size
        
        # Attention mechanism components
        self.query = nn.Linear(embedding_size, attention_size, bias=False)
        self.key = nn.Linear(embedding_size, attention_size, bias=False)
        self.value = nn.Linear(embedding_size, attention_size, bias=False)
        

    def forward(self, x):
        # x: [batch_size, 1080, 1]
        x = x.unsqueeze(-1)
        # Step 1: Embedding
        x = self.embedding(x)  # [batch_size, 1080, 128]
        
        # Step 2: Apply self-attention
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Calculate scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.attention_size ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.bmm(attention_weights, values)  # [batch_size, 1080, 128]
        
        # Step 3: Aggregation (Average pooling over the sequence)
        aggregated_features = torch.mean(attention_output, dim=1)  # [batch_size, 128]
        

        return aggregated_features



# class SelfAttention(nn.Module):
#     def __init__(self, input_size, attention_size):
#         super(SelfAttention, self).__init__()
#         self.in_features = input_size
#         self.attention_size = attention_size
        
#         self.query = nn.Linear(input_size, attention_size, bias=False)
#         self.key = nn.Linear(input_size, attention_size, bias=False)
#         self.value = nn.Linear(input_size, attention_size, bias=False)
#         self._norm_fact = 1 / math.sqrt(self.attention_size)

#     def forward(self, x):
#         # Compute queries, keys, and values
#         x = x.unsqueeze(1) # (batch_size, 1, input_size)
        
#         queries = self.query(x)
#         keys = self.key(x)
#         values = self.value(x)

#         dist = torch.bmm(queries, keys.transpose(1, 2)) * self._norm_fact
#         attention_weights = torch.softmax(dist, dim=-1)
    
#         out = torch.bmm(attention_weights, values).squeeze(1)

#         print(f"distribution: {dist.shape}") 
#         print(f"out: {out.shape}")
#         print(f"attention: {attention_weights.shape}")
#         # if self.is_plotting:
#             # np.save("data-analysis/data/attention/attention_scores.npy", attention_weights.squeeze().detach().numpy())
        
#         return out

