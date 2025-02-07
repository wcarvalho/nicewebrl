import jax
import jax.numpy as jnp
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np
from models.rnn_policy import ActorCriticRNN, ScannedRNN
from envs.environment import MiniGridEnv
import optax
import yaml
import pickle
import time

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_test_env(env_name):
    # Create environment with full observation
    env = gym.make(env_name, render_mode="human")
    env = FullyObsWrapper(env)
    return env

def calculate_reward(old_pos, new_pos, action, goal_pos=(4, 4)):
    """Calculate reward using the same structure as training"""
    # Convert positions to numpy arrays for easier calculation
    old_pos = np.array(old_pos)
    new_pos = np.array(new_pos)
    goal_pos = np.array(goal_pos)
    
    # Calculate rewards
    moved = action == 2
    
    # Check if moved toward goal
    old_dist = np.sqrt(np.sum((old_pos - goal_pos) ** 2))
    new_dist = np.sqrt(np.sum((new_pos - goal_pos) ** 2))
    moved_toward_goal = old_dist > new_dist
    
    # Reward structure
    at_goal = np.all(new_pos == goal_pos)
    goal_reward = 10.0 if at_goal else 0.0
    progress_reward = 0.5 if (moved_toward_goal and moved) else 0.0
    movement_reward = 0.1 if moved else -0.1
    
    return goal_reward + progress_reward + movement_reward

def test_agent(env, params, config):
    """Test the trained agent."""
    network = ActorCriticRNN(config=config, action_dim=2)
    hidden = ScannedRNN.initialize_carry(1, 128)
    
    # Create test environment with rendering
    test_env = gym.make(config["ENV_NAME"], render_mode="human")
    test_env = FullyObsWrapper(test_env)
    
    # Action mapping for our new 2-action space
    action_names = {
        0: "MOVE_TOWARD_GOAL",
        1: "RANDOM_MOVE"
    }
    
    total_rewards = []
    
    for episode in range(10):  # Run 10 test episodes
        timestep = env.reset()  # Just get TimeStep object
        obs = timestep.observation
        done = jnp.zeros((4,), dtype=bool)  # Initialize done as JAX array
        episode_reward = 0
        steps = 0
        
        # Reset render environment
        test_env.reset()
        
        while not done[0] and steps < 100:  # Check first environment's done flag
            # Process observation - reshape to match expected dimensions
            obs_array = obs[0:1]  # Take first batch element and keep dimensions
            dones_array = jnp.zeros((1, 1))
            
            # Get action from policy
            hidden, pi, _ = network.apply(params, hidden, (obs_array, dones_array))
            action = int(pi.mode()[0])
            
            # Take step
            print(f"\nStep {steps}:")
            print(f"Position: {timestep.state.position[0]}")
            print(f"Taking action: {action} ({action_names[action]})")
            
            # Execute action in both environments
            timestep, done = env.step(None, jnp.array([action]), timestep.state)
            test_env.step(action)  # Step render environment
            
            obs = timestep.observation
            reward = float(timestep.reward[0])
            
            print(f"New position: {timestep.state.position[0]}")
            print(f"Reward: {reward}")
            
            episode_reward += reward
            steps += 1
            
            # Add delay for visualization
            time.sleep(0.5)
        
        total_rewards.append(episode_reward)
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Steps taken: {steps}")
        print(f"Total reward: {episode_reward}")
    
    test_env.close()
    print(f"\nAverage reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards

def main():
    # Load config from yaml
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Override NUM_ENVS for testing
    config["NUM_ENVS"] = 1  # We only need one environment for testing
    
    env = MiniGridEnv(config=config, env_name=config["ENV_NAME"])
    network = ActorCriticRNN(config=config, action_dim=env.action_space)

    # Load model
    with open("checkpoints/final_model.pkl", "rb") as f:
        params = pickle.load(f)

    test_agent(env, params, config)

if __name__ == "__main__":
    main() 