#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation


from rich import print

class ActorCriticLidarCnn(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        num_lidar_scans=1081,
        kernel_size=3,
        out_channels=32,
        init_noise_std=1.0,
        **kwargs,
    ):
        # Calculate new LiDAR output size
        new_length = num_lidar_scans - kernel_size + 1  # New length after 1D convolution
        lidar_output_size = out_channels * new_length  # Total flattened size after convolution

        # Calculate the remaining data size
        remaining_data_size = num_actor_obs - num_lidar_scans  # Non-LiDAR observations

        # Calculate total observation size for actor and critic
        total_actor_obs = lidar_output_size + remaining_data_size
        total_critic_obs = total_actor_obs

        super().__init__(
            num_actor_obs=total_actor_obs,
            num_critic_obs=total_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        
        activation = get_activation(activation)

        self.lidar_cnn_a = LidarCnn(input_size=num_actor_obs, num_lidar_scans=num_lidar_scans, kernel_size=kernel_size, out_channels=out_channels)
        self.lidar_cnn_c = LidarCnn(input_size=num_critic_obs, num_lidar_scans=num_lidar_scans, kernel_size=kernel_size, out_channels=out_channels)
        
        self.actor.insert(0, self.lidar_cnn_a)
        self.critic.insert(0, self.lidar_cnn_c)
        
        print(f"Actor CNN: {self.actor}")
        print(f"Critic CNN: {self.critic}")

    def reset(self, dones=None):
        pass
    
    def act(self, observations, **kwargs):
        return super().act(observations=observations)

    def act_inference(self, observations):
        return super().act_inference(observations=observations)

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(critic_observations=critic_observations)




# class LidarCnn(nn.Module):
#     def __init__(self, num_lidar_scans, kernel_size, out_channels, num_layers=2):
#         super().__init__()
#         self.num_lidar_scans = num_lidar_scans

#         # Define initial convolutional layers
#         conv_layers = []

#         # First convolution layer
#         conv_layers.append(nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size))
#         conv_layers.append(nn.ReLU())  # Add activation function
        
#         # Additional convolutional layers with activations
#         for _ in range(1, num_layers):
#             conv_layers.append(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size))
#             conv_layers.append(nn.ReLU())  # Add activation function after each convolution
        
#         # Optional: add normalization or pooling
#         # conv_layers.append(nn.BatchNorm1d(out_channels))
#         # conv_layers.append(nn.MaxPool1d(kernel_size=2))

#         # Create the convolutional pipeline
#         self.conv_pipeline = nn.Sequential(*conv_layers)  # Correct the expected list elements

#     def forward(self, x):
#         # Prepare data for convolution
#         x_lidar = x[:, :self.num_lidar_scans].unsqueeze(1)  # (batch_size, 1, num_lidar_scans)

#         # Apply the convolutional pipeline
#         x_lidar = self.conv_pipeline(x_lidar)  # Apply all convolutional layers
        
#         # Flatten the 3D tensor into 2D
#         x_lidar = x_lidar.view(x_lidar.size(0), -1)  # Flatten to match concatenation requirements
        
#         # Concatenate with other data
#         x_remaining = x[:, self.num_lidar_scans:]
        
#         # Concatenate along the feature dimension
#         x = torch.cat([x_lidar, x_remaining], dim=1)  # Ensure compatible shape
        
#         return x









class LidarCnn(torch.nn.Module):
    def __init__(self, input_size, num_lidar_scans, kernel_size, out_channels):
        super().__init__()
        self.in_features = input_size
        self.num_lidar_scans = num_lidar_scans
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Ensure x has the expected shape for Conv1d
        x_lidar = x[:, :self.num_lidar_scans].unsqueeze(1)  # (batch_size, 1, num_lidar_scans)
        
        # Apply convolution
        x_lidar = self.conv1d(x_lidar)  # (batch_size, out_channels, new_length)
        x_lidar = self.activation(x_lidar)
        
        # Flatten the 3D tensor into 2D
        x_lidar = x_lidar.view(x_lidar.size(0), -1)  # (batch_size, out_channels * new_length)
        
        # Remaining data
        x_remaining = x[:, self.num_lidar_scans:]  # (batch_size, remaining_data_size)
        
        # Concatenate along the feature dimension
        x = torch.cat([x_lidar, x_remaining], dim=1)  # (batch_size, total_features)
        
        return x


    def reset(self, dones=None):
        pass