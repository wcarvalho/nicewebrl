"""JAX Compatible version of Catch bsuite environment.


Source: github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py.

Stolen from: https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/bsuite/catch.py
"""

from typing import Any, Dict, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp


@struct.dataclass
class EnvState:
  ball_x: chex.Array
  ball_y: chex.Array
  paddle_x: int
  paddle_y: int
  prev_done: bool
  time: int


@struct.dataclass
class EnvParams:
  max_steps_in_episode: int = 1000


class Catch:
  """JAX Compatible version of Catch bsuite environment."""

  def __init__(self, rows: int = 10, columns: int = 5):
    super().__init__()
    self.rows = rows
    self.columns = columns

  @property
  def default_params(self) -> EnvParams:
    # Default environment parameters
    return EnvParams()

  def step(
    self,
    key: chex.PRNGKey,
    state: EnvState,
    action: Union[int, float, chex.Array],
    params: EnvParams,
  ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
    """Perform single timestep state transition."""
    # Sample new init state each step & use if there was a reset!
    ball_x, ball_y, paddle_x, paddle_y = sample_init_state(key, self.rows, self.columns)
    prev_done = state.prev_done

    # Move the paddle + drop the ball.
    dx = action - 1  # [-1, 0, 1] = Left, no-op, right.
    paddle_x = jax.lax.select(
      prev_done,
      paddle_x,
      jnp.clip(state.paddle_x + dx, 0, self.columns - 1),
    )
    ball_y = jax.lax.select(prev_done, ball_y, state.ball_y + 1)
    ball_x = jax.lax.select(prev_done, ball_x, state.ball_x)
    paddle_y = jax.lax.select(prev_done, paddle_y, state.paddle_y)

    # Rewrite reward as boolean multiplication
    prev_done = ball_y == paddle_y
    catched = paddle_x == ball_x
    reward = prev_done * jax.lax.select(catched, 1.0, -1.0)

    state = state.replace(
      ball_x=ball_x,
      ball_y=ball_y,
      paddle_x=paddle_x,
      paddle_y=paddle_y,
      prev_done=prev_done,
      time=state.time + 1,
    )

    # Check number of steps in episode termination condition
    done = self.is_terminal(state, params)
    return (
      lax.stop_gradient(self.get_obs(state)),
      lax.stop_gradient(state),
      reward,
      done,
      {"discount": self.discount(state, params)},
    )

  def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
    """Reset environment state by sampling initial position."""
    ball_x, ball_y, paddle_x, paddle_y = sample_init_state(key, self.rows, self.columns)
    # Last two state vector correspond to timestep and done
    state = EnvState(
      ball_x=ball_x,
      ball_y=ball_y,
      paddle_x=paddle_x,
      paddle_y=paddle_y,
      prev_done=False,
      time=0,
    )
    return self.get_obs(state), state

  def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
    """Return observation from raw state trafo."""
    obs = jnp.zeros((self.rows, self.columns))
    obs = obs.at[state.ball_y, state.ball_x].set(1.0)
    obs = obs.at[state.paddle_y, state.paddle_x].set(1.0)
    return obs

  def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Check whether state is terminal."""
    done_loose = state.ball_y == self.rows - 1
    done_steps = state.time >= params.max_steps_in_episode
    done = jnp.logical_or(done_loose, done_steps)
    return done

  def discount(self, state, params) -> jnp.ndarray:
    """Return a discount of zero if the episode has terminated."""
    return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

  @property
  def name(self) -> str:
    """Environment name."""
    return "Catch-bsuite"

  @property
  def num_actions(self) -> int:
    """Number of actions possible in environment."""
    return 3

def render(state: EnvState, nrows: int, ncols: int):
  """Return a numpy array representing the game state."""
  # Create a blank canvas
  canvas = jnp.zeros((nrows, ncols))
  # Add paddle (value of 0.5)
  canvas = canvas.at[state.paddle_y, state.paddle_x].set(0.5)
  # Add ball (value of 1.0)
  canvas = canvas.at[state.ball_y, state.ball_x].set(1.0)
  return canvas


def sample_init_state(
  key: chex.PRNGKey, rows: int, columns: int
) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
  """Sample a new initial state."""
  ball_x = jax.random.randint(key, shape=(), minval=0, maxval=columns)
  ball_y = 0
  paddle_x = columns // 2
  paddle_y = rows - 1
  return ball_x, jnp.array(ball_y), paddle_x, paddle_y
