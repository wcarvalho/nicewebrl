from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

from flax import struct

import xminigrid
from xminigrid.experimental.img_obs import _render_obs


import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger


logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


########################################
# Define actions and corresponding keys
########################################
# actions defined here: https://github.com/dunnolab/xland-minigrid/blob/a46e78ce92f28bc90b8aac96d3b7b7792fb5bf3b/src/xminigrid/core/actions.py#L112


class Actions(IntEnum):
  FORWARD = 0
  RIGHT = 1
  LEFT = 2
  PICKUP = 3
  PUTDOWN = 4
  TOGGLE = 5


actions = [
  Actions.FORWARD,
  Actions.RIGHT,
  Actions.LEFT,
  Actions.PICKUP,
  Actions.PUTDOWN,
  Actions.TOGGLE,
]

action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowUp", "ArrowRight", "ArrowLeft", "p", "l", " "]
action_to_name = [a.name for a in actions]


########################################
# Define Craftax environment
########################################
# make environment
class FixXlandArgsWrapper:
  """Making reset and step functions match nicewebrl.JaxWebEnv"""

  def __init__(self, env):
    self._env = env

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def reset(self, key: jax.Array, params: struct.PyTreeNode) -> nicewebrl.TimeStep:
    return self._env.reset(params=params, key=key)

  def step(
    self,
    key: jax.Array,
    prior_timestep: nicewebrl.TimeStep,
    action: jax.Array,
    params: struct.PyTreeNode,
  ) -> nicewebrl.TimeStep:
    del key
    return self._env.step(params=params, timestep=prior_timestep, action=action)


# to list available environments: xminigrid.registered_environments()
jax_env, env_params = xminigrid.make("XLand-MiniGrid-R9-25x25")

key = jax.random.key(0)
reset_key, ruleset_key = jax.random.split(key)

# to list available benchmarks: xminigrid.registered_benchmarks()
benchmark = xminigrid.load_benchmark(name="trivial-1m")
# choosing ruleset, see section on rules and goals
ruleset = benchmark.sample_ruleset(ruleset_key)
env_params = env_params.replace(ruleset=ruleset)


# create web environment wrapper
jax_env = FixXlandArgsWrapper(jax_env)
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=env_params)


# Define rendering function
def render_fn(timestep: nicewebrl.TimeStep):
  image = _render_obs(timestep.observation)
  return image.astype(jnp.uint8)


# jit it so fast
render_fn = jax.jit(render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, env_params)


########################################
# Define Stages of experiment
########################################
all_stages = []


# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """
          - Press the arrow keys to move the agent
          - Press the space bar to 'toggle' objects
          - Press 'p' to pickup objects and 'l' to put them down
          """
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


# ------------------
# Environment stage
# ------------------


def make_image_html(src):
  html = f"""
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 200%; height: 200%; object-fit: contain;">
  </div>
  """
  return html


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data("stage_state")

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

    # --------------------------------
    # display environment
    # --------------------------------
    ui.html(make_image_html(src=state_image))


def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
  """Episode finishes if reward observated"""
  success = timestep.reward.sum() > 0
  return success


environment_stage = EnvStage(
  name="Environment",
  web_env=jax_web_env,
  action_keys=action_keys,
  action_to_name=action_to_name,
  env_params=env_params,
  render_fn=render_fn,
  vmap_render_fn=vmap_render_fn,
  display_fn=env_stage_display_fn,
  autoreset_on_done=True,
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
