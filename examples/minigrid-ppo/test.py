import jax
import jax.numpy as jnp
import numpy as np
import navix as nx
from flax.core import freeze
from simulation_2 import ActorCritic
from wrappers import LogWrapper, FlattenObservationWrapper, NavixEnvWrapper
from enum import IntEnum
import time
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

# Define actions as in experiment_structure.py
class Actions(IntEnum):
    # Only turn left, turn right, move forward are used in empty environment
    left = 0
    right = 1
    forward = 2
    # These actions are unused but kept for compatibility
    unused1 = 3  
    unused2 = 4
    unused3 = 5
    unused4 = 6

action_to_name = ["Left", "Right", "Forward", "Unused1", "Unused2", "Unused3", "Unused4"]

def main():
    print("Starting visualization - look for a matplotlib window...")
    print("(If no window appears, try moving your terminal window)")
    
    # Create base environment and check observation shape
    base_env = NavixEnvWrapper("Navix-Empty-8x8-v0", observation_fn=nx.observations.rgb)
    base_obs, _ = base_env.reset(jax.random.PRNGKey(0))
    print(f"Base observation shape: {base_obs.shape}")
    
    # Add wrappers
    env = LogWrapper(base_env)
    
    # Get input dimension and check shape
    obs_sample, _ = env.reset(jax.random.PRNGKey(0))
    print(f"Observation shape: {obs_sample.shape}")
    
    # Load model parameters and check their shapes
    params_numpy = np.load("updated_model_params.npy", allow_pickle=True).item()
    print("\nModel parameter shapes:")
    print(jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, params_numpy))
    params = freeze(params_numpy)
    
    # Create network with same action dimension as training
    network = ActorCritic(
        action_dim=env.action_space.n,
        activation="relu"
    )
    
    # Run episodes
    num_episodes = 5
    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} ===")
            
            obs, env_state = env.reset(jax.random.PRNGKey(episode))
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                # Get action from policy
                pi, value = network.apply(params, obs)
                action_rng = jax.random.PRNGKey(step)
                action = pi.sample(seed=action_rng)
                action = jnp.asarray(action).item()
                
                # Clip action to only use meaningful actions
                action = min(action, 2)  # Only LEFT, RIGHT, FORWARD
                
                # Take step
                obs, env_state, reward, done, info = env.step(
                    jax.random.PRNGKey(step), 
                    env_state, 
                    action
                )
                total_reward += reward
                
                # Print info using proper action names
                action_name = action_to_name[int(action)]
                print(f"Step {step}: Action={action_name}, Reward={reward}")
                
                # Render
                env.render(obs)
                time.sleep(0.5)
                
                step += 1
                
            print(f"Episode {episode + 1} finished with reward {total_reward} in {step} steps")

    finally:
        plt.ioff()  # Turn off interactive mode
        plt.close('all')

if __name__ == "__main__":
    main()
