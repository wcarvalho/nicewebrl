import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict
from envs.environment import MiniGridEnv
from models.rnn_policy import ActorCriticRNN, ScannedRNN
import os
import pickle
import distrax


class Transition(NamedTuple):
    """Transition class for storing rollouts"""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    done: jnp.ndarray
    hidden_state: jnp.ndarray


def compute_gae(traj_batch, last_val, config):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0
    for t in reversed(range(len(traj_batch.reward))):
        delta = (
            traj_batch.reward[t]
            + config["GAMMA"] * traj_batch.done[t] * last_val
            - traj_batch.value[t]
        )
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * traj_batch.done[t] * gae
        advantages.insert(0, gae)
        last_val = traj_batch.value[t]
    advantages = jnp.array(advantages)
    returns = advantages + traj_batch.value
    return advantages, returns


def make_train(config):
    """Create training function with given config"""
    env = MiniGridEnv(config=config, env_name=config["ENV_NAME"])
    
    # Verify action space matches config
    assert env.action_space == config["NUM_ACTIONS"], f"Environment action space {env.action_space} doesn't match config NUM_ACTIONS {config['NUM_ACTIONS']}"

    def train(rng):
        # Initialize actor-critic network
        network = ActorCriticRNN(config=config, action_dim=env.action_space)
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros((config["NUM_ENVS"], *env.observation_space))
        init_dones = jnp.zeros((config["NUM_ENVS"],))
        init_hidden = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        params = network.init(init_rng, init_hidden, (init_obs, init_dones))

        # Calculate number of updates
        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
        print(f"Will perform {num_updates} updates")

        # Initialize optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"])
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)

        # Environment reset
        timestep = env.reset()
        hidden = init_hidden

        def rollout(state, _):
            """Collect single rollout step"""
            train_state, hidden, timestep, rng = state
            rng, action_rng = jax.random.split(rng)

            # Forward pass through the actor-critic
            obs = timestep.observation
            hidden, pi, value = network.apply(train_state.params, hidden, (obs, timestep.last()))
            action = pi.sample(seed=action_rng)
            log_prob = pi.log_prob(action)

            # Step environment
            next_timestep, done = env.step(None, action, timestep.state)
            transition = Transition(
                obs=obs,
                action=action,
                reward=next_timestep.reward,
                value=value.reshape(-1),
                log_prob=log_prob,
                done=timestep.last(),
                hidden_state=hidden
            )
            return (train_state, hidden, next_timestep, rng), transition

        def update_step(state, _):
            """Perform single update step"""
            train_state, hidden, last_timestep, rng = state
            
            # Get last value for GAE
            _, _, last_value = network.apply(
                train_state.params, 
                hidden, 
                (last_timestep.observation, last_timestep.last())
            )

            # Collect rollouts
            (train_state, hidden, next_timestep, rng), traj_batch = jax.lax.scan(
                rollout, (train_state, hidden, last_timestep, rng), None, config["NUM_STEPS"]
            )

            # Compute GAE and returns
            advantages, returns = compute_gae(traj_batch, last_value.reshape(-1), config)

            # Process minibatches
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            permutation = jax.random.permutation(rng, batch_size)
            
            for start in range(0, batch_size, config["MINIBATCH_SIZE"]):
                end = min(start + config["MINIBATCH_SIZE"], batch_size)
                mb_inds = permutation[start:end]
                
                mb_batch = Transition(
                    obs=traj_batch.obs[mb_inds],
                    action=traj_batch.action[mb_inds],
                    reward=traj_batch.reward[mb_inds],
                    value=traj_batch.value[mb_inds],
                    log_prob=traj_batch.log_prob[mb_inds],
                    done=traj_batch.done[mb_inds],
                    hidden_state=traj_batch.hidden_state[mb_inds]
                )
                
                train_state, _ = update_ppo(
                    train_state,
                    mb_batch,
                    advantages[mb_inds],
                    returns[mb_inds],
                    rng
                )

            return (train_state, hidden, next_timestep, rng), None

        # Training loop
        state = (train_state, hidden, timestep, rng)
        for update in range(num_updates):
            state, _ = update_step(state, None)
            if update % 10 == 0:
                print(f"Update {update}/{num_updates} complete")

        # Save final model - simplified to one file
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/final_model.pkl", "wb") as f:
            pickle.dump(state[0].params, f)
        print("Training complete. Model saved as 'final_model.pkl'")

        return state[0]

    return train


def update_ppo(train_state, traj_batch, advantages, returns, rng):
    """Update policy and value function"""
    def loss_fn(params):
        mb_size = traj_batch.obs.shape[0]
        hidden_size = traj_batch.hidden_state[0].shape[-1]
        mb_hidden = jnp.zeros((mb_size, hidden_size))
        
        # Get policy and value predictions
        _, pi, v = train_state.apply_fn(
            params, 
            mb_hidden,
            (traj_batch.obs, traj_batch.done)
        )
        
        # Handle logits shape
        logits = pi.logits
        if len(logits.shape) > 2:
            logits = logits.reshape(-1, logits.shape[-1])
        logits = logits[:mb_size]
        pi = distrax.Categorical(logits=logits)
        
        # Handle action and old_log_prob shapes
        action = traj_batch.action[:, 0] if len(traj_batch.action.shape) > 1 else traj_batch.action
        old_log_prob = traj_batch.log_prob[:, 0] if len(traj_batch.log_prob.shape) > 1 else traj_batch.log_prob
        
        # Handle advantages and returns shapes
        advantages_reshaped = advantages[:, 0] if len(advantages.shape) > 1 else advantages
        returns_reshaped = returns[:, 0] if len(returns.shape) > 1 else returns
        
        # Get value predictions for the minibatch
        value_pred = v.squeeze()[:mb_size]
        
        # Print shapes only once at the start of training
        if rng[0] == 0:
            print("\nInitial tensor shapes:")
            print(f"Action: {action.shape}")
            print(f"Old log prob: {old_log_prob.shape}")
            print(f"Logits: {logits.shape}")
            print(f"Advantages: {advantages_reshaped.shape}")
            print(f"Returns: {returns_reshaped.shape}")
            print(f"Value pred: {value_pred.shape}\n")
        
        # Policy loss calculation
        new_log_prob = pi.log_prob(action)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        clip_eps = 0.2
        
        policy_loss1 = advantages_reshaped * ratio
        policy_loss2 = advantages_reshaped * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(policy_loss1, policy_loss2).mean()
        
        # Value loss with corrected shapes
        value_loss = 0.5 * ((value_pred - returns_reshaped) ** 2).mean()
        
        # Entropy loss
        entropy_loss = -0.01 * pi.entropy().mean()
        
        total_loss = policy_loss + value_loss + entropy_loss
        return total_loss, (policy_loss, value_loss, entropy_loss)

    # Compute gradients and update
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    metrics = {
        "total_loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
    }

    return train_state, metrics
