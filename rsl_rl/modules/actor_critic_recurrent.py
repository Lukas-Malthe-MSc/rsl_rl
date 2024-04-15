#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories


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
        attention=False,
        attention_type=None,
        attention_dims=None,
        attention_heads=None,
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

        print("**************************************************************************************")
        print("* ATTENTION TIME | ATTENTION TIME | ATTENTION TIME | ATTENTION TIME | ATTENTION TIME *")
        print("**************************************************************************************")

        self.use_attention = attention
        activation = get_activation(activation)

        if self.use_attention:
            assert attention_dims is not None, "Attention dimensions must be provided if attention is enabled"
            self.cross_attention = CrossAttention(
                query_dim=attention_dims[0],
                key_dim=attention_dims[1],
                value_dim=attention_dims[2],
                num_heads=attention_heads,
            )

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        if self.use_attention:
            print(f"Attention: {self.cross_attention}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        # input_a = self.memory_a(observations, masks, hidden_states)
        # return super().act(input_a.squeeze(0))
        actor_hidden = self.memory_a(observations, masks, hidden_states)
        if self.use_attention:
            # Assume critic_hidden is available or needs to be generated in a similar fashion to actor_hidden
            critic_hidden = self.memory_c(observations, masks, hidden_states)  # This might need context-specific adjustment
            actor_hidden = self.cross_attention(actor_hidden, critic_hidden, critic_hidden)

        action_output = super().act(actor_hidden.squeeze(0))
        return action_output

   
    def act_inference(self, observations):
        # input_a = self.memory_a(observations)
        # return super().act_inference(input_a.squeeze(0))
        actor_hidden = self.memory_a(observations)
        if self.use_attention:
            critic_hidden = self.memory_c(observations)
            actor_hidden = self.cross_attention(actor_hidden, critic_hidden, critic_hidden)

        action_output = super().act_inference(actor_hidden.squeeze(0))
        return action_output

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # input_c = self.memory_c(critic_observations, masks, hidden_states)
        # return super().evaluate(input_c.squeeze(0))
        critic_hidden = self.memory_c(critic_observations, masks, hidden_states)
        if self.use_attention:
            actor_hidden = self.memory_a(critic_observations, masks, hidden_states)
            critic_hidden = self.cross_attention(critic_hidden, actor_hidden, actor_hidden)
        
        value_output = super().evaluate(critic_hidden.squeeze(0))
        return value_output


    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=1):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, key_dim)
        self.value = nn.Linear(value_dim, value_dim)
        self.out = nn.Linear(value_dim, query_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        _attention = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        _attention = torch.nn.functional.softmax(_attention, dim=-1)
        out = torch.matmul(_attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.query_dim)
        return self.out(out)