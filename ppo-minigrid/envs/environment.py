from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
from enum import IntEnum
from functools import partial


# Define step types
class StepType(IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


# Define environment state as an immutable PyTree
@struct.dataclass
class EnvState:
    position: jnp.ndarray  # Shape (B, 2)
    direction: jnp.ndarray  # Shape (B,)
    step_count: jnp.ndarray
    max_steps: jnp.ndarray


# Define time step
class TimeStep(struct.PyTreeNode):
    state: EnvState
    observation: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    step_type: int

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


# MiniGrid Wrapper
class MiniGridEnv:
    def __init__(self, config, env_name="MiniGrid-Empty-5x5-v0"):
        self.env = FullyObsWrapper(gym.make(env_name, render_mode="rgb_array"))
        self.observation_space = self.env.observation_space["image"].shape
        self.action_space = self.env.action_space.n  # This line determines the action space
        print(f"Environment action space: {self.action_space}")
        print(f"Config NUM_ACTIONS: {config['NUM_ACTIONS']}")
        self.max_steps = self.env.unwrapped.max_steps
        self.config = config
        self.num_envs = config["NUM_ENVS"]
        self.goal_pos = jnp.array([4, 4])  # Goal position

        # Define MiniGrid actions
        self.TURN_LEFT = 0
        self.TURN_RIGHT = 1
        self.MOVE_FORWARD = 2
        self.PICKUP = 3
        self.DROP = 4
        self.TOGGLE = 5
        self.DONE = 6

    def reset(self, params=None):
        obs, info = self.env.reset()
        obs_image = jnp.array(obs["image"], dtype=jnp.float32) / 255.0
        obs_image = jnp.repeat(obs_image[None, ...], self.num_envs, axis=0)

        # Initialize state with batch dimension
        init_pos = jnp.repeat(jnp.array(self.env.unwrapped.agent_pos, dtype=jnp.int32)[None, :], self.num_envs, axis=0)
        init_dir = jnp.repeat(jnp.array([self.env.unwrapped.agent_dir], dtype=jnp.int32), self.num_envs)

        state = EnvState(
            position=init_pos,
            direction=init_dir,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            max_steps=jnp.ones(self.num_envs, dtype=jnp.int32) * self.max_steps,
        )

        return TimeStep(
            state=state,
            observation=obs_image,
            reward=jnp.zeros(self.num_envs),
            discount=jnp.ones(self.num_envs),
            step_type=jnp.array([int(StepType.FIRST)] * self.num_envs),
        )

    def step(self, params, action, prior_state: EnvState):
        """Pure JAX implementation for training."""
        # Ensure action matches batch size
        assert action.shape[0] == self.num_envs, "Action batch size mismatch!"

        # Get current position and direction
        pos_f32 = prior_state.position.astype(jnp.float32)
        current_direction = prior_state.direction

        # Direction vectors for each cardinal direction (N, E, S, W)
        direction_vectors = jnp.array([
            [0, -1],  # North (0)
            [1, 0],   # East  (1)
            [0, 1],   # South (2)
            [-1, 0],  # West  (3)
        ])

        # Get current direction vector
        current_dir_vec = direction_vectors[current_direction]

        # Update direction based on turn actions
        new_direction = jnp.where(
            action == self.TURN_LEFT,
            (current_direction - 1) % 4,
            jnp.where(
                action == self.TURN_RIGHT,
                (current_direction + 1) % 4,
                current_direction
            )
        )

        # Determine movement
        move_vec = jnp.where(
            action[:, None] == self.MOVE_FORWARD,
            current_dir_vec,
            jnp.zeros_like(current_dir_vec)  # No movement for other actions
        )

        # Update position
        new_position = jnp.clip(
            prior_state.position + jnp.round(move_vec).astype(jnp.int32),
            0,
            self.observation_space[0] - 1,
        )

        # Calculate distance to goal for reward
        goal_pos = jnp.array([self.goal_pos] * self.num_envs)
        old_dist = jnp.sqrt(jnp.sum((pos_f32 - goal_pos) ** 2, axis=1))
        new_dist = jnp.sqrt(jnp.sum((new_position.astype(jnp.float32) - goal_pos) ** 2, axis=1))

        # Check if at goal
        at_goal = jnp.all(new_position == goal_pos, axis=1)
        moved_toward_goal = new_dist < old_dist

        # Reward calculation
        reward = jnp.where(
            at_goal,
            50.0,  # Large reward for reaching the goal
            jnp.where(
                action == self.MOVE_FORWARD,
                jnp.where(
                    moved_toward_goal,
                    5.0,  # Reward for progress
                    -2.0,  # Penalty for regress
                ),
                jnp.where(
                    (action == self.TURN_LEFT) | (action == self.TURN_RIGHT),
                    -1.0,  # Small penalty for turning
                    -5.0,  # Larger penalty for other actions
                )
            )
        )

        # Update state
        new_state = EnvState(
            position=new_position,
            direction=new_direction,
            step_count=prior_state.step_count + 1,
            max_steps=prior_state.max_steps,
        )

        # Determine if episode is done
        done = (new_state.step_count >= new_state.max_steps) | at_goal | (action == self.DONE)

        timestep = TimeStep(
            state=new_state,
            observation=self._get_obs(new_state, goal_pos),
            reward=reward,
            discount=jnp.where(done, 0.0, 1.0),
            step_type=jnp.where(done, StepType.LAST, StepType.MID),
        )

        return timestep, done

    def _get_obs(self, state, goal_pos):
        """Create observation with position and vector to goal."""
        obs = jnp.zeros((self.num_envs, *self.observation_space), dtype=jnp.float32)

        vec_to_goal = goal_pos - state.position
        dist_to_goal = jnp.sqrt(jnp.sum(vec_to_goal ** 2, axis=1))
        normalized_vec = vec_to_goal / (dist_to_goal[:, None] + 1e-8)

        # Fill observation channels
        obs = obs.at[..., 0].set(state.position[:, 0, None, None] / self.observation_space[0])
        obs = obs.at[..., 1].set(state.position[:, 1, None, None] / self.observation_space[1])
        obs = obs.at[..., 2].set(normalized_vec[:, 0, None, None])  # Direction to goal

        return obs

    def step_real(self, action):
        """Real environment step for testing/evaluation."""
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = terminated or truncated
        return obs, reward, done, info
