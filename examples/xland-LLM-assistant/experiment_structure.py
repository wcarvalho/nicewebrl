from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from xminigrid.types import RuleSet
from xminigrid.benchmarks import (
  Benchmark,
  load_benchmark,
  load_benchmark_from_path,
  load_bz2_pickle,
  DATA_PATH,
  NAME2HFFILENAME,
)
from xminigrid.rendering.text_render import print_ruleset

# utils for the demonstation
from xminigrid.core.grid import room
from xminigrid.types import AgentState
from xminigrid.core.actions import take_action
from xminigrid.core.constants import Tiles, Colors, TILES_REGISTRY
from xminigrid.rendering.rgb_render import render
from xminigrid.benchmarks import Benchmark

# rules and goals
from xminigrid.core.goals import check_goal, AgentNearGoal
from xminigrid.core.rules import check_rule, AgentNearRule

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


########################################
# Define actions and corresponding keys
########################################
class Actions(IntEnum):
  forward = 0
  right = 1
  left = 2
  pick_up = 3
  put_down = 4
  toggle = 5


# Only first 3 actions are actually used
actions = jnp.array([0, 1, 2, 3, 4, 5])
action_keys = ["ArrowUp", "ArrowRight", "ArrowLeft", "p", "d", "t"]  # Mapping to keys
action_to_name = ["Forward", "Right", "Left", "Pick Up", "Drop", "Toggle"]

########################################
# Create multiple environments with different rulesets
########################################


def describe_goal(encoding):
  # Convert JAX arrays to Python integers
  goal_type = int(encoding[0])
  obj1_type, obj1_color = int(encoding[1]), int(encoding[2])
  obj2_type, obj2_color = int(encoding[3]), int(encoding[4])

  # Define mappings from integers to human-readable strings
  tile_names = {
    0: "empty space",
    1: "floor",
    2: "wall",
    3: "ball",
    4: "square",
    5: "pyramid",
    6: "goal",
    7: "key",
    8: "locked door",
    9: "closed door",
    10: "open door",
    11: "hex",
    12: "star",
  }

  color_names = {
    0: "",
    1: "red",
    2: "green",
    3: "blue",
    4: "purple",
    5: "yellow",
    6: "grey",
    7: "black",
    8: "orange",
    9: "white",
    10: "brown",
    11: "pink",
  }

  def describe_object(tile_type, color):
    name = tile_names.get(tile_type, "unknown")
    color_name = color_names.get(color, "")
    return f"{color_name} {name}".strip()

  obj1_desc = describe_object(obj1_type, obj1_color)
  obj2_desc = describe_object(obj2_type, obj2_color)

  # Define natural language descriptions for different goal types
  goal_descriptions = {
    0: "no goal (padding)",
    1: f"have the agent hold the {obj1_desc}",
    2: f"move the agent onto the {obj1_desc}",
    3: f"move the agent near the {obj1_desc}",
    4: f"move the {obj1_desc} near the {obj2_desc}",
    5: f"move the {obj1_desc} onto position of the {obj2_desc}",
    6: f"move the agent to the position of the {obj1_desc}",
    7: f"move the {obj1_desc} above the {obj2_desc}",
    8: f"move the {obj1_desc} to the right of the {obj2_desc}",
    9: f"move the {obj1_desc} below the {obj2_desc}",
    10: f"move the {obj1_desc} to the left of the {obj2_desc}",
    11: f"move the agent above the {obj1_desc}",
    12: f"move the agent to the right of the {obj1_desc}",
    13: f"move the agent below the {obj1_desc}",
    14: f"move the agent to the left of the {obj1_desc}",
  }

  return goal_descriptions.get(goal_type, "Unknown goal")


def create_env_with_ruleset(ruleset_key):
  env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
  benchmark = xminigrid.load_benchmark(name="medium-1m")
  rule = benchmark.sample_ruleset(jax.random.key(ruleset_key))
  rule_text = describe_goal(rule.goal)
  env_params = env_params.replace(ruleset=rule)
  env = GymAutoResetWrapper(env)
  env = RGBImgObservationWrapper(env)
  return env, env_params, rule_text


num_envs = 1

# Create 5 different environments
envs_and_params_and_ruletext = [create_env_with_ruleset(i) for i in range(num_envs)]


class PlaygroundTimestepWrapper(TimestepWrapper):
  def reset(self, key: jax.random.PRNGKey, params=None):
    timestep = self._env.reset(key=key, params=params)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(128, 128, 3), method="bilinear"
    ).astype(jnp.uint8)
    return TimeStep(
      state=timestep.replace(observation=resized_obs),
      observation=resized_obs,
      discount=jnp.ones((), dtype=jnp.float32),
      reward=jnp.zeros((), dtype=jnp.float32),
      step_type=jnp.array(0, dtype=jnp.uint8),
    )

  def step(self, key, state, action, params=None):
    if isinstance(state, TimeStep):
      state = state.state
    timestep = self._env.step(params=params, timestep=state, action=action)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(128, 128, 3), method="bilinear"
    ).astype(jnp.uint8)
    return TimeStep(
      state=timestep.replace(observation=resized_obs),
      observation=resized_obs,
      discount=jnp.ones((), dtype=jnp.float32),
      reward=timestep.reward,
      step_type=jnp.where(
        timestep.step_type == 2,
        jnp.array(2, dtype=jnp.uint8),  # LAST
        jnp.array(1, dtype=jnp.uint8),  # MID
      ),
    )


# Create JaxWebEnv for each environment
jax_web_envs = []
for env, env_params, rule_text in envs_and_params_and_ruletext:
  jax_env = PlaygroundTimestepWrapper(env, autoreset=True, use_params=True)
  jax_web_env = JaxWebEnv(
    env=jax_env,
    actions=actions,
  )
  jax_web_env.precompile(dummy_env_params=env_params)
  jax_web_envs.append((jax_web_env, env_params))


def render_fn(timestep: nicewebrl.TimeStep):
  return timestep.observation.astype(jnp.uint8)


render_fn = jax.jit(render_fn)

########################################
# Define Stages of experiment
########################################

all_stages = []


async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """Press the arrows keys to move the agent 
      
      Press:
      - p to pick up an object
      - d to drop an object
      - t to transform an object"""
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


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
  print("Display function obs shape:", timestep.observation.shape)
  rendered_img = stage.render_fn(timestep)
  new_obs_base64 = base64_npimage(rendered_img)

  stage_state = stage.get_user_data("stage_state")
  rule_text = stage.metadata.get("rule_text", "No rule text provided.")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )
    ui.markdown(f"**Goal:** {rule_text}")
    ui.html(make_image_html(src=new_obs_base64))


def evaluate_success_fn(timestep: TimeStep, params: Optional[object] = None):
  return timestep.last() and timestep.reward > 0


# Create 5 different stages
for i, (jax_web_env, env_params) in enumerate(jax_web_envs):
  # Get the rule_text for the current environment
  # envs_and_params_and_ruletext is a list of (env, env_params, rule_text)
  current_rule_text = envs_and_params_and_ruletext[i][2]

  environment_stage = EnvStage(
    name=f"Environment {i + 1}",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=jax_web_env.precompile_vmap_render_fn(render_fn, env_params),
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    msg_display_time=2,
    metadata=dict(
      desc=f"XLand environment {i + 1}",
      stage_number=i + 1,
      rule_text=current_rule_text,
    ),
  )
  all_stages.append(environment_stage)

experiment = nicewebrl.SimpleExperiment(
  stages=all_stages,
  randomize=[False] + [True] * (len(all_stages) - 1),
  )