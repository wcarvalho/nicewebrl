from enum import IntEnum
from typing import Optional, Tuple, Union, Any
import asyncio
import time
import jax
import jax.numpy as jnp
from nicegui import ui, app

from flax import struct


from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl import make

import nicewebrl
from nicewebrl import base64_npimage
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
from nicewebrl import broadcast_message
from asyncio import Lock
from nicewebrl.multihuman_stages import MultiHumanLeaderFollowerEnvStage
import rendering

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1

#####################################
# Helper functions
#####################################
_user_locks = {}
global_lock = Lock()

# This is used to ensure that each user has a unique lock


def get_user_lock():
  """A function that returns a lock for the current user using their unique seed"""
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


def args_from_room(args):
  user_room = str(app.storage.user.get("room_id", None))
  if user_room != str(args["called_by_room_id"]):
    return False
  return True


########################################
# Multi-agent Timestep Wrapper and Jax Web Environment
########################################


class MultiAgentTimestepWrapper(object):
  """

  Assumptions:
  1. Rewards are symmetric
  2. All agents end at the same time
  """

  def __init__(
    self,
    env,
    autoreset: bool = True,
    use_params: bool = False,
    num_leading_dims: int = 3,
  ):
    self._env = env
    self._autoreset = autoreset
    self._use_params = use_params
    self._num_leading_dims = num_leading_dims

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def reset(
    self, key: jax.random.PRNGKey, params: Optional[struct.PyTreeNode] = None
  ) -> Tuple[nicewebrl.TimeStep, dict]:
    if self._use_params:
      obs, state = self._env.reset(key, params)
    else:
      obs, state = self._env.reset(key)
    # Get shape from first leaf of obs, assuming it's a batch dimension
    first_leaf = jax.tree_util.tree_leaves(obs)[0]
    shape = first_leaf.shape[self._num_leading_dims :]
    timestep = nicewebrl.TimeStep(
      state=state,
      observation=obs,
      discount=jnp.ones(shape, dtype=jnp.float32),
      reward=jnp.zeros(shape, dtype=jnp.float32),
      step_type=jnp.full(
        shape, nicewebrl.StepType.FIRST, dtype=nicewebrl.StepType.FIRST.dtype
      ),
    )
    return timestep

  def step(
    self,
    key: jax.random.PRNGKey,
    prior_timestep: nicewebrl.TimeStep,
    action: Union[int, float],
    params: Optional[struct.PyTreeNode] = None,
  ) -> Tuple[nicewebrl.TimeStep, dict]:
    def env_step(prior_timestep_):
      if self._use_params:
        obs, state, reward, done, info = self._env.step(
          key, prior_timestep_.state, action, params
        )
      else:
        obs, state, reward, done, info = self._env.step(
          key, prior_timestep_.state, action
        )

      if type(done) is dict:
        # NOTE: assumes global done signal
        done = done["__all__"]
      if type(reward) is dict:
        # NOTE: assumes symmetric rewards
        reward = reward["agent_0"].astype(jnp.float32)

      del info
      return nicewebrl.TimeStep(
        state=state,
        observation=obs,
        discount=1.0 - done.astype(jnp.float32),
        reward=reward,
        step_type=jnp.where(done, nicewebrl.StepType.LAST, nicewebrl.StepType.MID),
      )

    if self._autoreset:
      # if prior was last, reset
      # otherwise, do regular step
      timestep = jax.lax.cond(
        prior_timestep.last(),
        lambda: self.reset(key, params),
        lambda: env_step(prior_timestep),
      )
    else:
      timestep = env_step(prior_timestep)
    return timestep


class MultiAgentJaxWebEnv:
  def __init__(self, env, actions):
    """The main purpose of this class is to precompile jax functions before experiment starts."""
    self.env = env
    assert hasattr(env, "reset"), "env needs reset function"
    assert hasattr(env, "step"), "env needs step function"

    def reset(rng, params):
      return env.reset(rng, params)

    def next_steps(rng, timestep, env_params):
      # vmap over rngs and actions. re-use timestep
      def step_env(rng, timestep, agent_0_action, agent_1_action, env_params):
        action_dict = {
          "agent_0": agent_0_action,
          "agent_1": agent_1_action,
        }
        return env.step(rng, timestep, action_dict, env_params)

      step_env = jax.vmap(step_env, in_axes=(None, None, 0, None, None), out_axes=0)

      step_env = jax.vmap(step_env, in_axes=(None, None, None, 0, None), out_axes=1)

      timesteps = step_env(rng, timestep, actions, actions, env_params)

      return timesteps

    self.reset = jax.jit(reset)
    self.next_steps = jax.jit(next_steps)

  def precompile(self, dummy_env_params: Optional[struct.PyTreeNode] = None) -> None:
    """Call this function to pre-compile jax functions before experiment starts."""
    logger.info("Compiling jax environment functions.")
    start = time.time()
    dummy_rng = jax.random.PRNGKey(0)
    self.reset = self.reset.lower(dummy_rng, dummy_env_params).compile()
    timestep = self.reset(dummy_rng, dummy_env_params)
    self.next_steps = self.next_steps.lower(
      dummy_rng, timestep, dummy_env_params
    ).compile()
    logger.info(f"\ttime: {time.time() - start}")

  def precompile_vmap_render_fn(
    self, render_fn, dummy_env_params: Optional[struct.PyTreeNode] = None
  ):
    """Call this function to pre-compile a multi-render function before experiment starts."""
    logger.info("Compiling multi-render function.")
    start = time.time()

    # vmap over actions from both agents
    vmap_render_fn = jax.jit(jax.vmap(jax.vmap(render_fn)))
    dummy_rng = jax.random.PRNGKey(0)
    timestep = self.reset(dummy_rng, dummy_env_params)
    next_timesteps = self.next_steps(dummy_rng, timestep, dummy_env_params)
    vmap_render_fn = vmap_render_fn.lower(next_timesteps).compile()
    logger.info(f"\ttime: {time.time() - start}")
    return vmap_render_fn


########################################
# Define actions and corresponding keys
########################################
class Actions(IntEnum):
  # Turn left, turn right, move forward
  up = 0
  down = 1
  right = 2
  left = 3
  stay = 4
  interact = 5
  done = 6


actions = [
  Actions.up,
  Actions.down,
  Actions.right,
  Actions.left,
  Actions.stay,
  Actions.interact,
]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowUp", "ArrowDown", "ArrowRight", "ArrowLeft", "s", " "]
action_to_name = [a.name for a in actions]

########################################
# Define Overcooked environment
########################################
# make environment
layout = overcooked_layouts["cramped_room"]
jax_env = make("overcooked", layout=layout, max_steps=MAX_EPISODE_TIMESTEPS)

# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
jax_env = MultiAgentTimestepWrapper(
  jax_env,
  autoreset=True,
  use_params=False,
  num_leading_dims=3,
)

# create web environment wrapper
jax_web_env = MultiAgentJaxWebEnv(env=jax_env, actions=action_array)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile()


# Define rendering function
def render_fn(timestep: nicewebrl.TimeStep):
  return rendering.render_fn(timestep.state)


# jit it so fast
render_fn = jax.jit(render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn)


########################################
# Define Stages of experiment
########################################
all_stages = []

# ------------------
# Instruction stage
# ------------------


async def get_room_stage_object(key: str, default: Optional[Any] = None) -> Any:
  """Retrieves a value from the room-stage specific storage system.

  This function retrieves values from a hierarchical storage system that maintains
  separate states for different combinations of rooms and stages.

  Args:
      key (str): The identifier for the value to retrieve.
      default (Optional[Any]): Value to return if the key doesn't exist. Defaults to None.

  Returns:
      Any: The stored value for the given key, or the default value if not found.

  Example:
      >>> # Get a dictionary tracking user completion
      >>> checked = get_room_stage_object('checked', {})
      >>> print(checked)  # {'user1': True, 'user2': False}

      >>> # Get a value with a default
      >>> score = get_room_stage_object('current_score', default=0)
  """
  async with get_user_lock():
    room_id = app.storage.user["room_id"]
    stage_idx = app.storage.user["stage_idx"]
    room_stage = f"{room_id}_{stage_idx}"
    all_room_objects = app.storage.general.get("room_objects", {})
    room_stage_objects = all_room_objects.get(room_stage, {})
    return room_stage_objects.get(key, default)


async def set_room_stage_object(key: str, value: Any) -> None:
  """Stores a value in the room-stage specific storage system.

  This function implements a hierarchical storage system that maintains separate
  states for different combinations of rooms and stages. It's primarily used
  for maintaining game state in a multi-room, multi-stage environment.

  Storage Structure:
      app.storage.general['room_objects'] = {
          'room1_stage0': {
              'key1': value1,
              'key2': value2
          },
          'room2_stage1': {
              'key1': value3,
              ...
          }
      }

  Args:
      key (str): The identifier for the value being stored.
      value (Any): The value to store. Can be any serializable object.

  Example:
      >>> # Store a dictionary tracking user completion
      >>> checked = {'user1': True, 'user2': False}
      >>> set_room_stage_object('checked', checked)

      >>> # Store a simple value
      >>> set_room_stage_object('current_score', 100)

  Note:
      - The room_id and stage_idx are automatically retrieved from app.storage.user
      - Values are stored in app.storage.general['room_objects']
      - Previous values for the same key will be overwritten
  """
  async with get_user_lock():
    room_id = app.storage.user["room_id"]
    stage_idx = app.storage.user["stage_idx"]
    room_stage = f"{room_id}_{stage_idx}"
    all_room_objects = app.storage.general.get("room_objects", {})
    room_stage_objects = all_room_objects.get(room_stage, {})
    room_stage_objects[key] = value
    all_room_objects[room_stage] = room_stage_objects
    app.storage.general["room_objects"] = all_room_objects


def get_room_users():
  room_id = app.storage.user["room_id"]
  return app.storage.general["rooms"][str(room_id)]


async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    logger.info(f"Cleared elements for stage {stage.name}")
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """
        - Press the arrow keys to move the agent
        - Press the space bar to interact with objects
        """
    )
    agent_color = "red" if app.storage.user["leader"] else "blue"
    ui.html(
      f'you will control the <span style="color: {agent_color}">{agent_color}</span> agent'
    )

    room_users = get_room_users()
    ui.markdown("=" * 30)
    ui.markdown(
      """press a key when ready.
      when both participants press a key, you will advance to the next stage.
      """
    )
    ui.markdown("Participants in room:")
    labels = {}
    for r in room_users:
      labels[str(r)] = ui.label(f"{r}")

    broadcast_message("stage_over", "false")
    #########################################
    # Create key listener that sets user to ready
    #########################################

    async def handle_key_press(event, c):
      room_users = get_room_users()
      checked = await get_room_stage_object("checked", {r: False for r in room_users})
      user_id = str(app.storage.user["seed"])
      checked[user_id] = True
      await set_room_stage_object("checked", checked)
      logger.info(checked)
      if sum(checked.values()) == 2:
        logger.info("keypress: broadcasting stage over")
        # activate a function in human javascript file that will emit an
        broadcast_message("stage_over", "true")

    checked = await get_room_stage_object("checked", {r: False for r in room_users})
    stage.custom_key_press_fn = handle_key_press
    if sum(checked.values()) == 2:
      logger.info("diplay_fn: broadcasting stage over")
      broadcast_message("stage_over", "true")


instruction_stage = Stage(name="Instuctions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


# ------------------
# Environment stage
# ------------------


def make_image_html(src):
  html = f'''
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
  </div>
  '''
  return html


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: nicewebrl.TimeStep
):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_room_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed and how many successful
    # --------------------------------
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )

    agent_color = "red" if app.storage.user["leader"] else "blue"
    user_to_action_idx = app.storage.general["user_to_action_idx"]
    agent_idx = int(user_to_action_idx[str(app.storage.user["seed"])]) + 1
    ui.html(
      f'<span style="color: {agent_color}">Agent {agent_idx} color: {agent_color}</span>'
    )
    # --------------------------------
    # display environment
    # --------------------------------

    ui.html(make_image_html(src=state_image))


def evaluate_success_fn(timestep: nicewebrl.TimeStep):
  """Episode finishes if person every gets 1 achievement"""
  success = timestep.reward > 0.5
  return success


environment_stage = MultiHumanLeaderFollowerEnvStage(
  name="Environment",
  web_env=jax_web_env,
  action_keys=action_keys,
  action_to_name=action_to_name,
  render_fn=render_fn,
  vmap_render_fn=vmap_render_fn,
  display_fn=env_stage_display_fn,
  evaluate_success_fn=evaluate_success_fn,
  min_success=MIN_SUCCESS_EPISODES,
  max_episodes=MAX_STAGE_EPISODES,
  verbosity=VERBOSITY,
  # add custom metadata to be stored here
  metadata=dict(
    # nothing required, just for bookkeeping
    desc="some description",
    key1="value1",
    key2="value2",
  ),
)
all_stages.append(environment_stage)
