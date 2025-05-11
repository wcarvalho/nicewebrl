from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

from flax import struct
import navix as nx

import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
from shared_state import previous_obs_base64, current_obs_base64
logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1

########################################
# Define actions and corresponding keys
########################################
class Actions(IntEnum):
    # Only turn left, turn right, move forward are used in empty environment
    left = 0
    right = 1
    forward = 2
    # These actions are unused but kept for compatibility
    unused1 = 3  
    unused2 = 4
    unused3 = 5

# Only first 3 actions are actually used
actions = jnp.array([0, 1, 2])  # Minigrid actions (left, right, forward)
action_keys = ["ArrowLeft", "ArrowRight", "ArrowUp"]  # Mapping to keys
action_to_name = ["Left", "Right", "Forward"]

########################################
# Define Minigrid environment
########################################
rows = 8
cols = 8
# Create environment with RGB observations explicitly
jax_env = nx.make("Navix-Empty-8x8-v0", observation_fn=nx.observations.rgb)

# Create a struct for environment parameters
@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = MAX_EPISODE_TIMESTEPS

default_params = EnvParams()

# Create a custom wrapper to handle Navix Timestep conversion and resize observations
class NavixTimestepWrapper(TimestepWrapper):
    def reset(self, key: jax.random.PRNGKey, params=None):
        timestep = self._env.reset(key)
        # Resize observation from 256x256x3 to 128x128x3
        resized_obs = jax.image.resize(
            timestep.observation,
            shape=(128, 128, 3),  # Increased to 128x128 for more detail
            method='bilinear'  # Keep bilinear for smooth resizing
        ).astype(jnp.uint8)
        return TimeStep(
            state=timestep.replace(observation=resized_obs),
            observation=resized_obs,
            discount=jnp.ones((), dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            step_type=jnp.array(0, dtype=jnp.uint8)
        )

    def step(self, key, state, action, params=None):
        if isinstance(state, TimeStep):
            state = state.state
        timestep = self._env.step(state, action)
        # Resize observation from 256x256x3 to 128x128x3
        resized_obs = jax.image.resize(
            timestep.observation,
            shape=(128, 128, 3),  # Increased to 128x128 for more detail
            method='bilinear'  # Keep bilinear for smooth resizing
        ).astype(jnp.uint8)
        return TimeStep(
            state=timestep.replace(observation=resized_obs),
            observation=resized_obs,
            discount=jnp.ones((), dtype=jnp.float32),
            reward=timestep.reward,
            step_type=jnp.where(timestep.is_done(),
                              jnp.array(2, dtype=jnp.uint8),
                              jnp.array(1, dtype=jnp.uint8))
        )

# Wrap environment
jax_env = NavixTimestepWrapper(
    jax_env,
    autoreset=True,
    use_params=True
)

# Create web environment wrapper
jax_web_env = JaxWebEnv(
    env=jax_env,
    actions=actions,
)

# Add these debug prints
print("\nBEFORE WRAPPER:")
raw_obs = jax_env._env.reset(jax.random.PRNGKey(0)).observation
print("Raw observation shape:", raw_obs.shape)

print("\nAFTER WRAPPER:")
wrapped_obs = jax_env.reset(jax.random.PRNGKey(0)).observation
print("Wrapped observation shape:", wrapped_obs.shape)

# Add these debug prints right before precompilation
print("\nPRECOMPILE TIMESTEP CHECK:")
dummy_timestep = jax_env.reset(jax.random.PRNGKey(0))
print("dummy timestep.observation shape:", dummy_timestep.observation.shape)
print("dummy timestep.state.observation shape:", dummy_timestep.state.observation.shape)

print("\nPRECOMPILE:")
# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=default_params)

# Define rendering function that resizes for display
def render_fn(timestep: nicewebrl.TimeStep):
    """Render Minigrid observation as an RGB image."""
    # Resize 8x8x3 to 256x256x3 only for display
    return jax.image.resize(
        timestep.observation,
        shape=(256, 256, 3),
        method='nearest'
    ).astype(jnp.uint8)

# jit it so fast
render_fn = jax.jit(render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, default_params
)
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
    ui.markdown("""Press the arrows keys to move the agent and p,d,t to pick up, drop, and toggle objects"""
    )

instruction_stage = Stage(name="Instuctions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)

# ------------------
# Environment stage
# ------------------

env_params = default_params.replace(
    max_steps_in_episode=MAX_EPISODE_TIMESTEPS
)

def make_image_html(src):
  html = f"""
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 400px; height: 400px; object-fit: contain;">
  </div>
  """
  return html

async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
    global previous_obs_base64, current_obs_base64
    print("Display function obs shape:", timestep.observation.shape)
    rendered_img = stage.render_fn(timestep)
    new_obs_base64 = base64_npimage(rendered_img)

    previous_obs_base64 = current_obs_base64
    current_obs_base64 = new_obs_base64

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
        ui.html(make_image_html(src=new_obs_base64))

def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
  """Episode finishes if person gets 5 achievements"""
  return timestep.last() and timestep.reward > 0

environment_stage = EnvStage(
  name="Environment",
  web_env=jax_web_env,
  action_keys=action_keys,
  action_to_name=action_to_name,
  env_params=env_params,
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