from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from .input_attention import InputAttention


class ActorCriticInputAttention(ActorCritic):
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

        print("*********************************************************************************")
        print("* IO ATTENTION TIME | IO ATTENTION TIME | IO ATTENTION TIME | IO ATTENTION TIME *")
        print("*********************************************************************************")

        activation = get_activation(activation)

        self.attention = InputAttention(input_dim=num_actor_obs, hidden_dim=rnn_hidden_size, num_heads=4)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        # self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)


        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Attention: {self.attention}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):

        # print(f"shape before attention: {observations.shape}, {masks}, {hidden_states}")
        # attenede_inputs = self.attention(observations)
        # print(f"shape after attention: {attenede_inputs.shape}")


        actor_hidden = self.memory_a(observations, masks, hidden_states)

        print(f"shape after mem_a: {actor_hidden.shape}")
        action_output = super().act(actor_hidden.squeeze(0))

        return action_output

        """""""""

        # Apply attention to input observations
        print(f"Observations: {observations.shape}")
        attended_inputs = self.attention(observations)
        
        # Process attended inputs with LSTM (memory)
        input_a = self.memory_a(attended_inputs, masks, hidden_states)
        action_output = super().act(input_a.squeeze(0))
        return action_output
        """""""""

   
    # TODO: Update this function
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
        # Print shape of all inputs
        critic_hidden = self.memory_c(critic_observations, masks, hidden_states)

        value_output = super().evaluate(critic_hidden.squeeze(0))
        return value_output



        """
        attended_inputs = self.attention(critic_observations)
        input_c = self.memory_c(attended_inputs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
        """
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
            print("why did i get here")
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0

