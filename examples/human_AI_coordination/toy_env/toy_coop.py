from collections import OrderedDict
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from functools import partial
import pdb
class Actions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4

@struct.dataclass
class State: 
    agent_pos: chex.Array  # (2, 2) array for both agents' positions
    goal_pos: chex.Array   # (2, 2) array for both goals' positions
    time: int
    terminal: bool

class ToyCoop(MultiAgentEnv):
    """Simple 5x5 cooperative gridworld"""
    def __init__(self, max_steps: int = 100, random_reset: bool = False, debug: bool = False):
        super().__init__(num_agents=2)
        self.width = 5
        self.height = 5
        self.max_steps = max_steps
        self.random_reset = random_reset
        self.debug = debug
        self.action_set = jnp.array([
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
        ])
        self.agents = ["agent_0", "agent_1"]

        # Movement vectors for each action
        self.action_to_dir = jnp.array([
            [1, 0],   # right
            [0, 1],   # down
            [-1, 0],  # left
            [0, -1],  # up
            [0, 0],   # stay
        ])

        self.all_pos = jnp.array([[x, y] for x in range(self.width) for y in range(self.height)])


    def reset(self, key: chex.PRNGKey, params=None) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state."""
        key1, key2 = jax.random.split(key)
        
        og_locations = jnp.array([[0, 2], [4, 2], [2, 0], [2, 4]])

        # Randomly place agents and goals
        indices = jax.random.permutation(key1, len(self.all_pos))[:4]
        rand_agent_pos = self.all_pos[indices[:2]]
        rand_goal_pos = self.all_pos[indices[2:]]

        agent_pos = jnp.where(self.random_reset, rand_agent_pos, og_locations[:2])
        goal_default = jnp.where(self.debug, og_locations[:2], og_locations[2:])
        goal_pos = jnp.where(self.random_reset, rand_goal_pos, goal_default)

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            time=0,
            terminal=False
        )
        
        obs = self.get_obs(state)
        return obs, state


    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""
        
        acts = jnp.array([actions["agent_0"], actions["agent_1"]])
        next_state, reward = self.step_agents(key, state, acts)
        
        next_state = next_state.replace(time=state.time + 1)
        done = self.is_terminal(next_state)
        next_state = next_state.replace(terminal=done)
        
        obs = self.get_obs(next_state)
        rewards = {"agent_0": reward, "agent_1": reward}
        shaped_reward = {"agent_0": 0, "agent_1": 0}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}
        
        return obs, next_state, rewards, dones, {"shaped_reward": shaped_reward}


    def step_agents(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: chex.Array,
    ) -> Tuple[State, float]:
        """Update agent positions and calculate rewards."""
        # Calculate next positions
        next_pos = state.agent_pos + self.action_to_dir[actions]
        
        # Bound positions to grid
        next_pos = jnp.clip(next_pos, 0, self.width - 1)
        
        # Check if positions would collide
        would_collide = jnp.all(next_pos[0] == next_pos[1])
        
        next_pos = jnp.where(would_collide, state.agent_pos, next_pos)
        
        # Modified reward calculation
        on_goal = lambda x, y: jnp.all(x == y)
        
        # Check which goal each agent is on (if any)
        agent0_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[0], state.goal_pos)
        agent1_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[1], state.goal_pos)
        
        # Only give reward if agents are on different goals
        both_on_goals = jnp.logical_and(jnp.any(agent0_goal), jnp.any(agent1_goal))
        on_same_goal = jnp.any(jnp.logical_and(agent0_goal, agent1_goal))
        reward = jnp.float32(jnp.logical_and(both_on_goals, ~on_same_goal)) * 3
        
        return state.replace(agent_pos=next_pos), reward - 1  # step cost


    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Convert state into agent observations."""
        # 5x5x3 observation: agent 0 position, agent 1 position, goal positions
        obs = jnp.zeros((self.height, self.width, 3))
        
        # Set agent positions
        obs = obs.at[state.agent_pos[0, 1], state.agent_pos[0, 0], 0].set(1)
        obs = obs.at[state.agent_pos[1, 1], state.agent_pos[1, 0], 1].set(1)
        
        # Set goal positions
        obs_0 = obs.at[state.goal_pos[:, 1], state.goal_pos[:, 0], 2].set(1)

        obs_1 = obs_0.at[:, :, 0].set(obs_0[:, :, 1])
        obs_1 = obs_1.at[:, :, 1].set(obs_0[:, :, 0])  # swap agent 0 and 1

        # Set current agent indicator (channel 3)
        # obs_0 = obs.at[state.agent_pos[0, 1], state.agent_pos[0, 0], 3].set(1)  # Agent 0's perspective
        # obs_1 = obs.at[state.agent_pos[1, 1], state.agent_pos[1, 0], 3].set(1)  # Agent 1's perspective
        
        # Flatten the observation
        # obs_0 = obs_0.reshape(-1)
        # obs_1 = obs_1.reshape(-1)
        
        return {
            "agent_0": obs_0,
            "agent_1": obs_1
        }


    def is_terminal(self, state: State) -> bool:
        """Check if episode is done."""
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "ToyCoop"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.height, self.width, 3))
