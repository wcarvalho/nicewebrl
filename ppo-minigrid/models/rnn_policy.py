import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Dict

__all__ = ["ActorCriticRNN", "ScannedRNN"]


# Recurrent Neural Network with GRUCell
class ScannedRNN(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        obs, done = x
        
        # Get batch size from obs
        batch_size = obs.shape[0]
        
        # Reshape rnn_state if needed
        if len(rnn_state.shape) == 3:  # (1, 4, 128)
            rnn_state = rnn_state.reshape(-1, self.hidden_size)  # (4, 128)
        
        # Match batch dimensions
        if rnn_state.shape[0] != batch_size:
            rnn_state = jnp.broadcast_to(rnn_state[:1], (batch_size, self.hidden_size))
        
        # Reset RNN state when episode is done
        done = done.reshape(-1)  # Flatten to [B]
        rnn_state = jnp.where(done[:, None], jnp.zeros_like(rnn_state), rnn_state)
        
        # Run GRU cell
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, obs)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """
        Initialize GRU hidden state.
        Args:
            batch_size: Number of parallel environments
            hidden_size: GRU hidden size
        Returns:
            Zero-initialized GRU hidden state (batch_size, hidden_size)
        """
        return jnp.zeros((batch_size, hidden_size))


# Actor-Critic Model with GRU
class ActorCriticRNN(nn.Module):
    config: dict
    action_dim: int = 7  # Updated to match MiniGrid's 7 actions
    activation: str = "tanh"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        
        # Get batch dimensions from input
        if len(obs.shape) == 5:  # [T, B, H, W, C]
            T, B, H, W, C = obs.shape
            obs = obs.reshape(T * B, H * W * C)
        elif len(obs.shape) == 4:  # [B, H, W, C]
            B, H, W, C = obs.shape
            obs = obs.reshape(B, H * W * C)
        else:  # [H, W, C]
            H, W, C = obs.shape
            obs = obs.reshape(1, H * W * C)
        
        # Project to hidden size (use correct input size)
        embedding = nn.Dense(
            features=128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        
        # RNN layer
        rnn = ScannedRNN(hidden_size=128)
        hidden, x = rnn(hidden, (embedding, dones))
        
        # Actor head
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x)
        actor_mean = nn.Dense(
            features=self.action_dim,  # Will now output logits for 7 actions
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(x)
        pi = distrax.Categorical(logits=actor_mean)
        
        # Critic head
        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(x)
        
        return hidden, pi, value
