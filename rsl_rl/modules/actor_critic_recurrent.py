#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

from rich import print

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        actions = super().act(input_a.squeeze(0))
        return actions

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
        self.attention = MultiHeadSelfAttention(input_size=hidden_size, num_heads=4, attention_size=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Adding LayerNorm layer

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            attn_out, _ = self.attention(out, out, out)
            out = out + attn_out
            out = unpad_trajectories(out, masks)

        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            attn_out, _ = self.attention(out, out, out)
            out = out + attn_out

        out = self.layer_norm(out)

        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
            
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, num_heads, attention_size):
        super().__init__()
        assert attention_size % num_heads == 0, "Attention size must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_size = attention_size // num_heads
        
        # Linear transformations for query, key, and value for each head
        self.query_linear = nn.Linear(input_size, attention_size, bias=False)
        self.key_linear = nn.Linear(input_size, attention_size, bias=False)
        self.value_linear = nn.Linear(input_size, attention_size, bias=False)
        
        # Final linear transformation after concatenating heads
        self.output_linear = nn.Linear(attention_size, input_size)
        
        self.scale = 1.0 / (self.head_size ** 0.5)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations for query, key, and value for each head
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute scaled dot-product attention for each head
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Concatenate the outputs of all heads and apply final linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        output = self.output_linear(attn_output)
        
        return output, attn_weights