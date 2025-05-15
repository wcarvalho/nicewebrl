import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from wrappers import FlattenObservationWrapper, LogWrapper, NavixEnvWrapper
import numpy as nx
import navix as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Initialize wandb with more detailed config
config = {
    "LR": 2.5e-4,  # Back to standard PPO learning rate
    "NUM_ENVS": 4,  # Fewer environments for more focused learning
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": int(5e5),
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ENV_NAME": "Navix-Empty-8x8-v0",
    "BATCH_SIZE": 32,  # Smaller batch size
    "OBS_SIZE": 64,
}

# Calculate number of updates
config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])

wandb.init(
    project="ppo-navix",
    name=f"ppo_training_{int(time.time())}",
    config=config
)

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]
        
        # Normalize input
        x = x.astype(jnp.float32) / 255.0
        
        # CNN without pooling
        x = nn.Conv(features=16, kernel_size=(8, 8))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Actor head
        x1 = nn.Dense(features=64)(x)
        x1 = nn.relu(x1)
        actor_mean = nn.Dense(features=self.action_dim)(x1)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        x2 = nn.Dense(features=64)(x)
        x2 = nn.relu(x2)
        critic = nn.Dense(features=1)(x2)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Dict


def compute_gae(rewards, values, dones, next_value, config):
    """Compute Generalized Advantage Estimation (GAE)."""
    # First reshape all inputs to have consistent batch size
    num_steps = len(rewards) // config["BATCH_SIZE"]
    
    # Reshape to (num_steps, batch_size)
    rewards = rewards.reshape(num_steps, config["BATCH_SIZE"])
    values = values.reshape(num_steps, config["BATCH_SIZE"])
    dones = dones.reshape(num_steps, config["BATCH_SIZE"])
    
    # Handle next_value differently - broadcast it to match batch size
    next_value = jnp.broadcast_to(next_value, (config["BATCH_SIZE"],))
    next_value = next_value.reshape(1, -1)  # Make it (1, batch_size)
    
    # Compute deltas
    next_values = jnp.concatenate([values[1:], next_value])
    deltas = rewards + config["GAMMA"] * next_values * (1 - dones) - values
    
    # Initialize advantages array
    advantages = jnp.zeros_like(deltas)
    last_gae = jnp.zeros(config["BATCH_SIZE"])
    
    # Compute GAE
    for t in reversed(range(num_steps)):
        last_gae = deltas[t] + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - dones[t]) * last_gae
        advantages = advantages.at[t].set(last_gae)
    
    # Compute returns and reshape
    returns = advantages + values
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def update_model(train_state, batch, clip_eps):
    """PPO update step."""
    def loss_fn(params, batch):
        obs, actions, old_log_probs, advantages, returns = batch
        pi, values = train_state.apply_fn(params, obs)
        
        # Policy loss
        log_probs = pi.log_prob(actions)
        old_log_probs = old_log_probs.reshape(-1)  # Ensure (batch_size,)
        ratio = jnp.exp(log_probs - old_log_probs)
        
        # Ensure all tensors are 1D with shape (batch_size,)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        values = values.reshape(-1)
        
        # Policy loss
        policy_objective = ratio * advantages
        clipped_objective = jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * advantages
        policy_loss = -jnp.minimum(policy_objective, clipped_objective).mean()
        
        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Back to constant entropy coefficient
        entropy_loss = -config["ENT_COEF"] * pi.entropy().mean()
        
        total_loss = policy_loss + config["VF_COEF"] * value_loss + entropy_loss
        
        return total_loss, (policy_loss, value_loss, entropy_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(
        train_state.params, batch
    )
    
    train_state = train_state.apply_gradients(grads=grads)
    
    return train_state, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss
    }


def create_debug_image(obs, reward, value, advantage, step_count):
    """Create a debug image with metrics overlaid."""
    # Convert observation to numpy and ensure correct shape
    obs = np.array(obs)
    if len(obs.shape) == 4:  # Remove batch dimension if present
        obs = obs[0]
    
    # Convert JAX arrays to Python scalars
    reward = float(reward) if np.isscalar(reward) else float(reward.item())
    value = float(value) if np.isscalar(value) else float(value.item())
    advantage = float(advantage) if np.isscalar(advantage) else float(advantage.item())
    step_count = int(step_count) if np.isscalar(step_count) else int(step_count.item())
    
    # Create figure with explicit DPI setting
    dpi = 100
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(obs)
    
    # Add metrics as text overlay
    info_text = (
        f"Step: {step_count}\n"
        f"Reward: {reward:.2f}\n"
        f"Value: {value:.2f}\n"
        f"Advantage: {advantage:.2f}"
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    ax.axis('off')
    
    # Save figure to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Convert buffer to numpy array
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)[:, :, :3]  # Convert to RGB


def make_train(config):
    env = NavixEnvWrapper(config["ENV_NAME"], observation_fn=nx.observations.rgb)
    env = LogWrapper(env)

    def train(rng):
        # Initialize environment and model
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng)
        
        network = ActorCritic(
            action_dim=env.action_space.n,
            activation=config["ACTIVATION"]
        )
        
        # Initialize training state
        rng, init_rng = jax.random.split(rng)
        params = network.init(init_rng, obs[None])  # Add batch dimension
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5)
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        # Training loop
        episode_rewards = []
        best_mean_reward = float('-inf')
        
        for update in range(config["NUM_UPDATES"]):
            # Collect trajectories
            transitions = []
            episode_reward = 0
            
            for step in range(config["NUM_STEPS"]):
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs)
                action = pi.sample(seed=_rng)
                action = jnp.asarray(action).item()
                
                # Clip action to only use meaningful actions (0-2)
                action = min(action, 2)  # Only LEFT, RIGHT, FORWARD
                
                log_prob = pi.log_prob(action)
                next_obs, env_state, reward, done, info = env.step(_rng, env_state, action)
                
                # Log environment visualization every 50 steps
                if (update * config["NUM_STEPS"] + step) % 50 == 0:
                    _, next_value = network.apply(train_state.params, next_obs)
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    advantage = delta
                    
                    debug_image = create_debug_image(
                        obs,
                        reward,
                        value[0] if hasattr(value, 'shape') else value,
                        advantage[0] if hasattr(advantage, 'shape') else advantage,
                        update * config["NUM_STEPS"] + step
                    )
                    wandb.log({
                        "environment_render": wandb.Image(debug_image),
                        "step_reward": reward,
                        "step_value": float(value[0]) if hasattr(value, 'shape') else float(value),
                        "step_advantage": float(advantage[0]) if hasattr(advantage, 'shape') else float(advantage),
                        "global_step": update * config["NUM_STEPS"] + step,
                    })
                
                transitions.append(Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob.reshape(-1),
                    obs=obs,
                    next_obs=next_obs,
                    info=info
                ))
                
                episode_reward += reward
                
                if done:
                    print(f"Episode completed with reward: {episode_reward:.2f}")
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    rng, _rng = jax.random.split(rng)
                    obs, env_state = env.reset(_rng)
                else:
                    obs = next_obs
            
            # Process trajectories
            batch = {k: jnp.array([getattr(t, k) for t in transitions]) 
                    for k in transitions[0]._fields if k != 'info'}
            
            # Compute advantages and returns
            advantages, returns = compute_gae(
                batch['reward'],
                batch['value'],
                batch['done'],
                network.apply(train_state.params, batch['next_obs'][-1])[1],
                config
            )
            
            # Update model
            for epoch in range(config["UPDATE_EPOCHS"]):
                indices = np.random.permutation(len(batch['obs']))
                for start in range(0, len(indices), config["BATCH_SIZE"]):
                    end = start + config["BATCH_SIZE"]
                    mb_idx = indices[start:end]
                    
                    mini_batch = (
                        batch['obs'][mb_idx],
                        batch['action'][mb_idx],
                        batch['log_prob'][mb_idx],
                        advantages[mb_idx].reshape(-1),
                        returns[mb_idx].reshape(-1)
                    )
                    
                    train_state, metrics = update_model(
                        train_state, 
                        mini_batch,
                        config["CLIP_EPS"]
                    )
            
            # Log metrics
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-100:])
                
                # Log to wandb
                wandb.log({
                    "mean_reward": mean_reward,
                    "episode_reward": episode_rewards[-1],
                    "episode_length": batch['done'].sum(),
                    "update": update,
                    **metrics
                })
                
                # Print more frequent updates
                if update % 1 == 0:  # Print every update instead of every 10
                    print(f"\nUpdate {update}/{config['NUM_UPDATES']}")
                    print(f"Mean Reward: {mean_reward:.2f}")
                    print(f"Episodes: {len(episode_rewards)}")
                    print(f"Steps: {(update + 1) * config['NUM_STEPS']}")
                    print(f"Current Loss: {metrics['total_loss']:.3f}")

            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                params_numpy = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
                np.save("v2_updated_model_params.npy", params_numpy)
        
        return train_state

    return train


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    train_fn = make_train(config)
    final_state = train_fn(rng)
    
    # Save final model
    params_numpy = jax.tree_util.tree_map(lambda x: np.array(x), final_state.params)
    np.save("final_model_params.npy", params_numpy)
    
    wandb.finish()
