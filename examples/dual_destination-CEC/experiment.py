from dotenv import load_dotenv
from flax import struct
import jax
import jax.numpy as jnp
import os
import pickle

from toy_env import ToyCoop, Actions, State
from toy_env import render_fn as toy_coop_render_fn
# import jaxmarl.make as make
import jaxmarl

from nicegui import ui, app
import nicewebrl
from nicewebrl import MultiAgentJaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, MultiAgentEnvStage
from nicewebrl import get_logger
from actor_networks import ActorCriticRNN, ScannedRNN
import inspect

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get('VERBOSITY', 0))
DEBUG = int(os.environ.get('DEBUG', 0))
WORLD_SEED = int(os.environ.get('WORLD_SEED', 1))
NAME = os.environ.get('NAME', 'exp')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 101
MIN_SUCCESS_EPISODES = 100

def get_user_save_file_fn():
    return f'{DATA_DIR}/user={app.storage.user.get("seed")}_name={NAME}_debug={DEBUG}.json'


########################################
# Define actions and corresponding keys
########################################
actions = [Actions.up, Actions.down, Actions.left, Actions.right, Actions.stay]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "s"]
action_to_name = [a.name for a in actions]


def initialize_environment(config):
    def filter_kwargs(kwargs, class_):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(class_.__init__).parameters}
        return filtered_kwargs
    env = ToyCoop(**filter_kwargs(config["ENV_KWARGS"], ToyCoop))
    config["obs_dim"] = (5,5,3)
    config['layout_name'] = 'toy_coop'
    return env

import yaml
from pathlib import Path
base_config = yaml.safe_load(Path('misc/toyCoop_config.yaml').read_text())
base_config['ENV_NAME'] = 'ToyCoop'
base_config['ENV_KWARGS']['random_reset'] = False
base_config['ENV_KWARGS']['debug'] = False
base_config['ENV_KWARGS']['max_steps'] = MAX_EPISODE_TIMESTEPS
base_config['GRAPH_NET'] = True


########################################
# Define ToyCoop environment
########################################
jax_env = initialize_environment(base_config)
default_params = {'random_reset_fn': 0}


########################################
# Load agent models
########################################
agent_model = ActorCriticRNN(action_dim=len(actions), config=base_config)
agent_model_params_1 = pickle.load(open('model_params/model_params_1.pkl', 'rb'))['params']
agent_model_params_2 = pickle.load(open('model_params/model_params_2.pkl', 'rb'))['params']
# stack params of multiple models users could potentially play
agent_model_params = jax.tree_map(lambda *x: jnp.stack(x), agent_model_params_1, agent_model_params_2)
init_hidden_state_fn = lambda : ScannedRNN.initialize_carry(1, base_config['GRU_HIDDEN_DIM'])


# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
jax_env = TimestepWrapper(jax_env,
 autoreset=True,
 reset_w_batch_dim=False,
 use_params=False)

# create web environment wrapper
jax_web_env = MultiAgentJaxWebEnv(
    env=jax_env,
    actions=action_array)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=default_params)

# Define rendering function
def render_fn(timestep: nicewebrl.TimeStep):
    image = toy_coop_render_fn(timestep.state)
    return image.astype(jnp.uint8)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, default_params)

# compile it so fast
render_fn = jax.jit(render_fn).lower(
    jax_web_env.reset(jax.random.PRNGKey(0), default_params)).compile()


########################################
# Define Stages of experiment
########################################
all_stages = []

# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
    with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown("These are instructions")

instruction_stage = Stage(
    name="Instuctions",
    display_fn=instruction_display_fn)
all_stages.append(instruction_stage)

# ------------------
# Environment stage
# ------------------
env_params = default_params

def make_image_html(src):
    html = f'''
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
    </div>
    '''
    return html

async def env_stage_display_fn(
        stage: MultiAgentEnvStage,
        container: ui.element,
        timestep: nicewebrl.TimeStep):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data('stage_state')

  with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        # --------------------------------
        # tell person how many episodes completed
        # --------------------------------
        with ui.row():
            with ui.element('div').classes('p-2 bg-green-100'):
                ui.label().bind_text_from(
                    stage_state, 'nepisodes', lambda n: f"Try: {n}/{stage.max_episodes}")

        # --------------------------------
        # display environment
        # --------------------------------
        ui.html(make_image_html(src=state_image))

def evaluate_success_fn(timestep: nicewebrl.TimeStep, env_params: struct.PyTreeNode):
    """Episode finishes if person every gets 1 achievement"""
    # success = int(timestep.state.terminal)
    # return success
    return False


environment_stage = MultiAgentEnvStage(
    name="toy_coop",
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
    user_save_file_fn=get_user_save_file_fn,
    metadata=dict(
        # nothing required, just for bookkeeping
        desc="some description",
        key1="value1",
        key2="value2",
    ),
    model=agent_model,
    model_params=agent_model_params,
    init_hidden_state_fn=init_hidden_state_fn,
    max_timesteps=MAX_EPISODE_TIMESTEPS,
    human_id=None,  # will randomly shuffle human id,
    using_param_stack=True
)
all_stages.append(environment_stage)
