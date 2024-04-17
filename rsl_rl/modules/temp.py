import torch
import torch.nn as nn

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
        transformer_layers=3,
        transformer_dim=512,
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()))

        # Initialize base ActorCritic with adapted input dimensions (from transformers)
        super(ActorCriticTransformer, self).__init__(
            num_actor_obs=transformer_dim,  # output dim of transformer
            num_critic_obs=transformer_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        
        activation = get_activation(activation)
        self.is_transformer = True
        
        # Initialize transformers
        self.transformer_a = Transformer(num_actor_obs, transformer_dim, num_heads, transformer_layers, transformer_dim)
        self.transformer_c = Transformer(num_critic_obs, transformer_dim, num_heads, transformer_layers, transformer_dim)
        
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
        # Reset transformers if needed (likely not necessary for standard transformers)
        self.transformer_a.reset(dones)
        self.transformer_c.reset(dones)


class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(model_dim, output_dim)
        
        self.x_buffer = None

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.input_projection(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.output_projection(x)
        print(f"Output shape: {x.shape}")
        return x

    def reset(self, dones=None):
        # No state to reset in standard transformer, but method exists for interface compatibility
        pass
