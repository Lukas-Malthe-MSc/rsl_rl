from __future__ import annotations

import torch
import torch.nn as nn

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

        self.attention_a = SelfAttention(input_size=num_actor_obs, attention_size=512)
        self.attention_c = SelfAttention(input_size=num_critic_obs, attention_size=512)

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
    def __init__(self, input_size, attention_size):
        super(SelfAttention, self).__init__()
        self.attention_size = attention_size
        self.in_features = input_size
        self.query = nn.Linear(input_size, attention_size)
        self.key = nn.Linear(input_size, attention_size)
        self.value = nn.Linear(input_size, attention_size)
        self.layer_norm = nn.LayerNorm(attention_size)

    def forward(self, x):
        # Compute queries, keys, and values
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / (self.attention_size ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)
        
        out = self.layer_norm(attended_values)
        return out