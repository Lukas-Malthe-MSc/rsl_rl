#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

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
        lidar_cnn_out_dim=485,
        kernel_size=5,
        out_channels=32,
        init_noise_std=1.0,
        **kwargs,
    ):
        # Calculate new LiDAR output size

        super().__init__(
            num_actor_obs=lidar_cnn_out_dim,
            num_critic_obs=lidar_cnn_out_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        
        activation = get_activation(activation)

        self.lidar_cnn_a = LidarCnn(input_size=num_actor_obs, num_lidar_scans=num_lidar_scans, kernel_size=kernel_size, out_channels=out_channels, activation=activation)
        self.lidar_cnn_c = LidarCnn(input_size=num_critic_obs, num_lidar_scans=num_lidar_scans, kernel_size=kernel_size, out_channels=out_channels, activation=activation)
        
        self.actor.insert(0, self.lidar_cnn_a)
        self.critic.insert(0, self.lidar_cnn_c)
        
        print(f"Actor CNN: {self.actor}")
        print(f"Critic CNN: {self.critic}")
        
        total_params_actor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print(f"Total trainable parameters in Actor: {total_params_actor}")
        
        total_params_critic = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(f"Total trainable parameters in Critic: {total_params_critic}")

    def reset(self, dones=None):
        pass
    
    def act(self, observations, **kwargs):
        return super().act(observations=observations)

    def act_inference(self, observations):
        output = super().act_inference(observations=observations)
        
        # try:
        #     lidar= observations[:, :1081]
        #     activations = self.get_activations(observations)
        #     np.save("data-analysis/data/lidar_2.npy", lidar.cpu().numpy())
        #     np.save("data-analysis/data/activations_2.npy", np.array(activations, dtype=object))
        # except RuntimeError as e:
        #     print(f"Error generating activations: {e}")
        
        return output

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(critic_observations=critic_observations)


    def get_activations(self, observations):
        """Retrieve activations from the LiDAR CNN layers."""
        activations = []

        # Forward pass through the actor CNN
        x = observations[:, :1081].unsqueeze(1)  # (batch_size, 1, num_lidar_scans)
        for layer in self.lidar_cnn_a.pipeline:
            x = layer(x)
            activations.append(x.detach().squeeze(0).cpu().numpy())  # Store the activation
        
        return activations


class LidarCnn(nn.Module):
    def __init__(self, input_size, num_lidar_scans, kernel_size, out_channels, activation):
        super().__init__()
        self.num_lidar_scans = num_lidar_scans
        self.in_features = input_size
        # Define initial convolutional layers
        layers = []

        # First convolution layer
        layers.append(nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=2))
        layers.append(activation)  # Add activation function
        
        layers.append(nn.MaxPool1d(kernel_size=2))  # Add pooling layer
        
        layers.append(nn.Conv1d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=2))
        layers.append(activation)  # Add activation function
        
        layers.append(nn.MaxPool1d(kernel_size=2))  # Add pooling layer
        
        layers.append(nn.Conv1d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=kernel_size, stride=2))
        layers.append(activation)  # Add activation function
        
        layers.append(nn.MaxPool1d(kernel_size=2))  # Add pooling layer
        
        # Optional: add normalization or pooling
        # layers.append(nn.BatchNorm1d(out_channels))
        # layers.append(nn.MaxPool1d(kernel_size=2))

        # Create the convolutional pipeline
        self.pipeline = nn.Sequential(*layers)  # Correct the expected list elements

    def forward(self, x):
        # Prepare data for convolution
        x_lidar = x[:, :self.num_lidar_scans].unsqueeze(1)  # (batch_size, 1, num_lidar_scans)

        # Apply the convolutional pipeline
        x_lidar = self.pipeline(x_lidar)  # Apply all convolutional layers
        
        # Flatten the 3D tensor into 2D
        x_lidar = x_lidar.view(x_lidar.size(0), -1)  # Flatten to match concatenation requirements
        
        # Concatenate with other data
        x_remaining = x[:, self.num_lidar_scans:]
        
        # Concatenate along the feature dimension
        x = torch.cat([x_lidar, x_remaining], dim=1)  # Ensure compatible shape
        
        return x