from typing import Any, Callable, Dict, Optional
import time
import dataclasses
import base64

from flax import struct
from flax import serialization
import jax
import jax.numpy as jnp

from nicegui import app, ui
from nicewebrl import nicejax
from nicewebrl.nicejax import new_rng, base64_npimage, make_serializable
from tortoise import fields, models


def make_image_html(src):
    html = '<div id = "stateImageContainer" >'
    html += f'<img id = "stateImage" src="{src}">'
    html += '</div >'
    return html

class StageStateModel(models.Model):
    id = fields.IntField(primary_key=True)
    session_id = fields.CharField(
        max_length=255, index=True)  # Added max_length
    stage_idx = fields.IntField(index=True)
    data = fields.TextField()

    class Meta:
        table = "stage"

class ExperimentData(models.Model):
    id = fields.IntField(primary_key=True)
    session_id = fields.CharField(max_length=255, index=True)
    stage_idx = fields.IntField(index=True)
    image_seen_time = fields.TextField()
    action_taken_time = fields.TextField()
    computer_interaction = fields.TextField()
    action_name = fields.TextField()
    action_idx = fields.IntField(index=True)
    data = fields.TextField()

    class Meta:
        table = "experiment"

async def get_latest_stage_state(cls: struct.PyTreeNode) -> StageStateModel | None:
    latest = await StageStateModel.filter(
        session_id=app.storage.browser['id'],
        stage_idx=app.storage.user['stage_idx'],
    ).order_by('-id').first()

    if latest is not None:
        deserialized = nicejax.deserialize_bytes(cls, latest.data)
        return deserialized
    return latest

async def save_stage_state(stage_state):
    stage_state = jax.tree_map(make_serializable, stage_state)
    serialized_data = serialization.to_bytes(stage_state)
    encoded_data = base64.b64encode(serialized_data).decode('ascii')
    model = StageStateModel(
        session_id=app.storage.browser['id'],
        stage_idx=app.storage.user['stage_idx'],
        data=encoded_data,
    )
    await model.save()

@struct.dataclass
class EnvStageState:
    timestep: struct.PyTreeNode = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

@dataclasses.dataclass
class Stage:
    name: str = 'stage'
    body: str = 'stage'
    display_fn: Callable = None
    finished: bool = False

    def __post_init__(self):
        self.user_data = {}

    def get_user_data(self, key, value=None):
        user_seed = app.storage.user['seed']
        self.user_data[user_seed] = self.user_data.get(user_seed, {})
        return self.user_data[user_seed].get(key, value)

    def set_user_data(self, **kwargs):
        user_seed = app.storage.user['seed']
        self.user_data[user_seed] = self.user_data.get(user_seed, {})
        self.user_data[user_seed].update(kwargs)

    async def run(self, container: ui.element):
        self.display_fn(stage=self, container=container)
        button = ui.button('Next page')
        self.set_user_data(button=button)

    async def handle_button_press(self):
        self.set_user_data(finished=True)

    async def handle_key_press(self, e, container): pass

@dataclasses.dataclass
class EnvStage(Stage):
    instruction: str = 'instruction'
    max_episodes: Optional[int] = 10
    min_success: Optional[int] = 1
    web_env: Any = None
    env_params: struct.PyTreeNode = None
    render_fn: Callable = None
    vmap_render_fn: Callable = None
    evaluate_success_fn: Callable = lambda t: 0
    state_cls: EnvStageState = None
    action_to_key: Dict[int, str] = None
    action_to_name: Dict[int, str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.vmap_render_fn is None:
            self.vmap_render_fn = self.web_env.precompile_vmap_render_fn(
                self.render_fn, self.env_params)

        self.key_to_action = {k: a for a, k in self.action_to_key.items()}
        if self.action_to_name is None:
            self.action_to_name = dict()

    def step_and_send_timestep(self, container, timestep):
        start = time.time()
        #############################
        # get next images and store them client-side
        # setup code to display next wind
        #############################
        rng = new_rng()
        next_timesteps = self.web_env.next_steps(
            rng, timestep, self.env_params)
        print("- next_steps:", time.time()-start)
        start = time.time()
        next_images = self.vmap_render_fn(next_timesteps)

        next_images = {
            self.action_to_key[idx]: base64_npimage(image) for idx, image in enumerate(next_images)}

        js_code = f"window.next_states = {next_images};"
        ui.run_javascript(js_code)

        self.set_user_data(next_timesteps=next_timesteps)
        #############################
        # display image
        #############################
        self.display_fn(
            stage=self,
            container=container,
            timestep=timestep)
        ui.run_javascript("window.imageSeenTime = new Date();")


    async def start_stage(self, container: ui.element):
        rng = new_rng()
        timestep = self.web_env.reset(rng, self.env_params)
        stage_state = self.state_cls(timestep=timestep)
        self.set_user_data(
            #timestep=timestep,
            stage_state=stage_state,
        )
        await save_stage_state(stage_state)
        print('-'*10)
        print(f'{self.name}. start_stage')
        self.step_and_send_timestep(container, timestep)

    def load_stage(self, container: ui.element, stage_state: EnvStageState):
        rng = new_rng()
        timestep = nicejax.match_types(
            example=self.web_env.reset(rng, self.env_params),
            data=stage_state.timestep)
        self.set_user_data(
            stage_state=stage_state.replace(
                timestep=timestep),
        )
        print('-'*10)
        print(f'{self.name}. load_stage')
        self.step_and_send_timestep(container, timestep)

    async def run(self, container: ui.element):
        # (potentially) load stage state from memory
        stage_state = await get_latest_stage_state(
            cls=self.state_cls)

        if stage_state is None:
            await self.start_stage(container)
        else:
            self.load_stage(container, stage_state)


    async def save_experiment_data(self, javascript_inputs):

        key = javascript_inputs.args['key']
        keydownTime = javascript_inputs.args['keydownTime']
        imageSeenTime = javascript_inputs.args['imageSeenTime']
        action_idx = self.key_to_action[key]
        action_name = self.action_to_name.get(action_idx)

        timestep = self.get_user_data('stage_state').timestep
        timestep = jax.tree_map(make_serializable, timestep)
        serialized_timestep = serialization.to_bytes(timestep)
        encoded_timestep = base64.b64encode(
            serialized_timestep).decode('ascii')

        model = ExperimentData(
            session_id=app.storage.browser['id'],
            stage_idx=app.storage.user['stage_idx'],
            image_seen_time=imageSeenTime,
            action_taken_time=keydownTime,
            computer_interaction=key,
            action_name=action_name,
            action_idx=action_idx,
            data=encoded_timestep,
        )

        await model.save()

    async def handle_key_press(
            self,
            javascript_inputs,
            container):
        if self.get_user_data('finished', False): return

        key = javascript_inputs.args['key']
        # check if valid environment interaction
        if not key in self.key_to_action: return

        # save experiment data so far
        await self.save_experiment_data(javascript_inputs)

        # use action to select from avaialble next time-steps
        action_idx = self.key_to_action[key]
        next_timesteps = self.get_user_data('next_timesteps')
        timestep = jax.tree_map(
            lambda t: t[action_idx], next_timesteps)
        print('-'*10)
        print(f'{self.name}. handle_key_press')
        self.step_and_send_timestep(container, timestep)
        success = self.evaluate_success_fn(timestep)

        stage_state = self.get_user_data('stage_state')
        stage_state = stage_state.replace(
            timestep=timestep,
            nsteps=stage_state.nsteps + 1,
            nepisodes=stage_state.nepisodes + timestep.last(),
            nsuccesses=stage_state.nsuccesses + success,
        )
        await save_stage_state(stage_state)
        self.set_user_data(stage_state=stage_state)

        ################
        # Episode over?
        ################
        if timestep.last():
            if success:
                ui.notify(
                    'success', type='positive', position='center',
                    timeout=10,
                          )
            else:
                ui.notify(
                    'failure', type='negative', position='center',
                    timeout=10)

        ################
        # Stage over?
        ################
        achieved_min_success = stage_state.nsuccesses >= self.min_success
        achieved_max_episodes = stage_state.nepisodes >= self.max_episodes
        stage_finished = achieved_min_success or achieved_max_episodes
        self.set_user_data(finished=stage_finished)

