import torch
import torch.nn as nn
import math
from rich import print
from rsl_rl.modules.actor_critic import ActorCritic, get_activation
torch.set_printoptions(precision=2, threshold=5000, edgeitems=2, linewidth=150)

class ActorCriticTransformer(ActorCritic):
    is_transformer = True
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
        
        # Initialize transformers
        self.transformer_a = Transformer(num_actor_obs, transformer_dim, num_heads, transformer_layers, transformer_dim)
        self.transformer_c = Transformer(num_critic_obs, transformer_dim, num_heads, transformer_layers, transformer_dim)
        
        print(f"Actor Transformer: {self.transformer_a}")
        print(f"Critic Transformer: {self.transformer_c}")


    def act(self, observations, masks=None, **kwargs):
        input_a = self.transformer_a(observations, masks)
        return super().act(input_a)

    def act_inference(self, observations):
        input_a = self.transformer_a(observations)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, **kwargs):
        input_c = self.transformer_c(critic_observations, masks)
        return super().evaluate(input_c)
    
    def reset(self, dones=None):
        # Reset transformers if needed (likely not necessary for standard transformers)
        self.transformer_a.reset(dones)
        self.transformer_c.reset(dones)


class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_len=24):
        super(Transformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(model_dim, output_dim)
        
        self.max_seq_len = max_seq_len
        self.buffers = None

    def forward(self, x, masks=None):
        batch_mode = masks is not None

        # for testing puroposes
        if batch_mode:
            # Adjusting the shape of x for testing by slicing
            x = x[:, :32, :]  # This reduces the second dimension from 33 to 32

        if not batch_mode:
            if self.buffers is None:
                self.buffers = torch.zeros(self.max_seq_len, x.size(0), x.size(1), device=x.device)

            # Update buffers with new observations, shifting older data
            self.buffers = torch.roll(self.buffers, shifts=-1, dims=0)
            self.buffers[-1, :, :] = x  # Add new observation at the end of the buffer

            # Create masks based on valid data counts
            valid_data_counts = (self.buffers != 0).any(dim=2).long().sum(dim=0)
            masks = torch.arange(self.max_seq_len, device=x.device).expand(x.size(0), self.max_seq_len) >= valid_data_counts.unsqueeze(1)
            x = self.buffers

        # Project the input
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=masks) if not batch_mode else self.transformer_encoder(x)
        x = self.output_projection(x)
        x = torch.mean(x, dim=0)       # Average pooling across the timesteps, output shape [num_envs, output_dim]

        return x


    def reset(self, dones):
        # Reset buffers for environments that are done
        if self.buffers is not None and dones is not None:
            self.buffers[dones] = 0
            
            
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]