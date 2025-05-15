import warnings
warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import inspect
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import wandb
import functools
import pdb
import os
import pickle
from tqdm import tqdm

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree_map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["GRAPH_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            reshaped_obs = obs.reshape(-1, *self.config["obs_dim"])

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class ActorCriticE3T(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["GRAPH_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
 
            reshaped_obs = obs.reshape(-1, *self.config["obs_dim"])

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Model of other agent
        #########
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10)  # L2 normalization
        other_pi = distrax.Categorical(logits=prediction_other)

        #########
        # Actor
        #########
        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi