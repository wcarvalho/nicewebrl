
import asyncio
from typing import List,Tuple
import copy
from typing import Any, Callable, Dict, Optional
import time
import dataclasses
from datetime import datetime
import base64

from flax import struct
from flax import serialization
import jax
import jax.numpy as jnp

from nicegui import app, ui
from nicewebrl import nicejax
from nicewebrl.nicejax import new_rng, base64_npimage, make_serializable
from tortoise import fields, models


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
    session_id = fields.CharField(
        max_length=255, index=True)  # Added max_length
    stage_idx = fields.IntField(index=True)
    image_seen_time = fields.TextField()
    action_taken_time = fields.TextField()
    computer_interaction = fields.TextField()
    action_name = fields.TextField()
    action_idx = fields.IntField(index=True)
    user_data = fields.JSONField(default=dict, blank=True)
    metadata = fields.JSONField(default=dict, blank=True)
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
    asyncio.create_task(model.save())

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
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    display_fn: Callable = None
    finished: bool = False
    next_button: bool = True
    duration: int = None

    def __post_init__(self):
        self.user_data = {}

    def get_user_data(self, key, value=None):
        user_seed = app.storage.user['seed']
        self.user_data[user_seed] = self.user_data.get(user_seed, {})
        return self.user_data[user_seed].get(key, value)

    def pop_user_data(self, key, value=None):
        user_seed = app.storage.user['seed']
        self.user_data[user_seed] = self.user_data.get(user_seed, {})
        return self.user_data[user_seed].pop(key, value)

    def set_user_data(self, **kwargs):
        user_seed = app.storage.user['seed']
        self.user_data[user_seed] = self.user_data.get(user_seed, {})
        self.user_data[user_seed].update(kwargs)

    async def activate(self, container: ui.element):
        self.display_fn(stage=self, container=container)

    async def handle_button_press(self):
        # TODO: rename to "finish_stage"
        self.set_user_data(finished=True)

    async def finish_stage(self):
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
    reset_display_fn: Callable = None
    vmap_render_fn: Callable = None
    evaluate_success_fn: Callable = lambda t: 0
    check_finished: Callable = lambda t: False
    state_cls: EnvStageState = None
    action_to_key: Dict[int, str] = None
    action_to_name: Dict[int, str] = None
    next_button: bool = False
    notify_success: bool = True
    msg_display_time: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.vmap_render_fn is None:
            self.vmap_render_fn = self.web_env.precompile_vmap_render_fn(
                self.render_fn, self.env_params)

        self.key_to_action = {k: a for a, k in self.action_to_key.items()}
        if self.action_to_name is None:
            self.action_to_name = dict()

    def print(self, str):
        user_id = app.storage.user.get(
            'user_id', None)
        if user_id is not None:
            print(f"{user_id}:", str)
        else:
            print(str)

    def step_and_send_timestep(
            self,
            container,
            timestep,
            update_display: bool = True):
        #############################
        # get next images and store them client-side
        # setup code to display next wind
        #############################
        rng = new_rng()
        next_timesteps = self.web_env.next_steps(
            rng, timestep, self.env_params)
        next_images = self.vmap_render_fn(next_timesteps)

        next_images = {
            self.action_to_key[idx]: base64_npimage(image) for idx, image in enumerate(next_images)}

        js_code = f"window.next_states = {next_images};"

        ui.run_javascript(js_code)

        self.set_user_data(next_timesteps=next_timesteps)
        #############################
        # display image
        #############################
        if update_display:
            self.display_fn(
                stage=self,
                container=container,
                timestep=timestep,
                )
            ui.run_javascript("window.imageSeenTime = new Date();")
        else:
            ui.run_javascript(
                "window.imageSeenTime = window.next_imageSeenTime;")

    async def wait_for_start(
            self,
            container: ui.element,
            timestep: struct.PyTreeNode,
            ):
        ui.run_javascript("window.accept_keys = false;")
        if self.reset_display_fn is not None:
            await self.reset_display_fn(
                stage=self,
                container=container,
                timestep=timestep)

        ui.run_javascript("window.accept_keys = true;")

    async def start_stage(self, container: ui.element):
        rng = new_rng()

        # NEW EPISODE
        timestep = self.web_env.reset(rng, self.env_params)
        stage_state = self.state_cls(timestep=timestep).replace(
            nepisodes=1,
            nsteps=1,
        )
        self.set_user_data(stage_state=stage_state)
        asyncio.create_task(save_stage_state(stage_state))

        # DISPLAY NEW EPISODE
        await self.wait_for_start(container, timestep)
        self.step_and_send_timestep(container, timestep)


    async def load_stage(self, container: ui.element, stage_state: EnvStageState):
        rng = new_rng()
        timestep = nicejax.match_types(
            example=self.web_env.reset(rng, self.env_params),
            data=stage_state.timestep)
        self.set_user_data(stage_state=stage_state.replace(
            timestep=timestep),
        )
        #await self.wait_for_start(container, timestep)
        self.step_and_send_timestep(container, timestep)

    async def activate(self, container: ui.element):
        self.print("="*30)
        self.print(self.metadata)
        # (potentially) load stage state from memory
        stage_state = await get_latest_stage_state(
            cls=self.state_cls)

        if stage_state is None:
            await self.start_stage(container)
        else:
            await self.load_stage(container, stage_state)
        self.set_user_data(started=True)
        ui.run_javascript("window.accept_keys = true;")

    async def save_experiment_data(
            self,
            args,
            synchronous=True):
        key = args['key']
        keydownTime = args['keydownTime']
        imageSeenTime = args['imageSeenTime']
        action_idx = self.key_to_action.get(key, -1)
        action_name = self.action_to_name.get(
            action_idx, key)

        timestep = self.get_user_data('stage_state').timestep
        timestep = jax.tree_map(make_serializable, timestep)
        serialized_timestep = serialization.to_bytes(timestep)
        encoded_timestep = base64.b64encode(
            serialized_timestep).decode('ascii')

        step_metadata = copy.deepcopy(self.metadata)
        step_metadata.update(
            nsteps=int(self.get_user_data('stage_state').nsteps),
            episode_idx=int(self.get_user_data('stage_state').nepisodes),
            nsuccesses=int(self.get_user_data('stage_state').nsuccesses),
        )

        user_data = dict(
            user_id=app.storage.user['seed'],
            age=app.storage.user.get('age'),
            sex=app.storage.user.get('sex'),
        )

        model = ExperimentData(
            stage_idx=app.storage.user['stage_idx'],
            session_id=app.storage.browser['id'],
            image_seen_time=imageSeenTime,
            action_taken_time=keydownTime,
            computer_interaction=key,
            action_name=action_name,
            action_idx=action_idx,
            data=encoded_timestep,
            user_data=user_data,
            metadata=step_metadata,
        )
        if synchronous:
            await model.save()
        else:
            asyncio.create_task(model.save())

    async def finish_stage(self):
        if not self.get_user_data('started', False):
            return
        # save experiment data so far (prior time-step + resultant action)
        # if finished, save synchronously (to avoid race condition) with next stage
        self.set_user_data(finished_noreset=True)
        self.set_user_data(finished=True)
        imageSeenTime = await ui.run_javascript('getImageSeenTime()', timeout=10)

        start_notification = self.pop_user_data('start_notification')
        if start_notification:
            start_notification.dismiss()
        success_notification = self.pop_user_data('success_notification')
        if success_notification:
            success_notification.dismiss()

        await self.save_experiment_data(
            args=dict(
                key='timer',
                keydownTime=imageSeenTime,
                imageSeenTime=imageSeenTime,
            ),
            synchronous=True)

    async def handle_key_press(
            self,
            javascript_inputs,
            container):
        if not self.get_user_data('started', False):
            return

        self.print("-"*10)
        if self.get_user_data('finished', False):
            self.print("finished")
            return
        key = javascript_inputs.args['key']
        self.print(f'key: {key}')
        # check if valid environment interaction
        if not key in self.key_to_action: return

        # save experiment data so far (prior time-step + resultant action)
        # if finished, save synchronously (to avoid race condition) with next stage
        finished_noreset = self.get_user_data('finished_noreset', False)
        await self.save_experiment_data(
            javascript_inputs.args, synchronous=finished_noreset)

        # use action to select from avaialble next time-steps
        action_idx = self.key_to_action[key]
        next_timesteps = self.get_user_data('next_timesteps')
        timestep = jax.tree_map(
            lambda t: t[action_idx], next_timesteps)

        episode_reset = timestep.first()
        if episode_reset:
            start_notification = self.pop_user_data('start_notification')
            if start_notification: start_notification.dismiss()
            success_notification = self.pop_user_data('success_notification')
            if success_notification: success_notification.dismiss()

        success = self.evaluate_success_fn(timestep)

        stage_state = self.get_user_data('stage_state')
        stage_state = stage_state.replace(
            timestep=timestep,
            nsteps=stage_state.nsteps + 1,
            nepisodes=stage_state.nepisodes + timestep.first(),
            nsuccesses=stage_state.nsuccesses + success,
        )
        self.print(f"nsteps: {stage_state.nsteps}")
        self.print(f"nepisodes: {stage_state.nepisodes}")
        self.print(f"nsuccesses: {stage_state.nsuccesses}")

        if finished_noreset:
            await save_stage_state(stage_state)
        else:
            asyncio.create_task(save_stage_state(stage_state))
        self.set_user_data(stage_state=stage_state)

        ################
        # Stage over?
        ################
        achieved_min_success = stage_state.nsuccesses >= self.min_success
        achieved_max_episodes = stage_state.nepisodes > self.max_episodes
        finished = (achieved_min_success or achieved_max_episodes)
        finished = finished or self.check_finished(timestep)

        # stage is finished AFTER final time-step of last episode
        # i.e. once the episode resets
        stage_finished = episode_reset and finished
        self.set_user_data(finished_noreset=finished)
        self.set_user_data(finished=stage_finished)

        ################
        # Display new data?
        ################
        update_display = not stage_finished and episode_reset
        if update_display:
            await self.wait_for_start(container, timestep)
        self.step_and_send_timestep(
            container, timestep,
            # image is normally updated client-side
            # when episode resets, update server-side
            update_display=update_display,
            )
        ################
        # Episode over?
        ################
        if timestep.last():
            if not stage_finished:
                start_notification = ui.notification(
                    'press any arrow key to start next episode',
                    position='center', type='info', timeout=self.msg_display_time)
            success_notification = None
            if self.notify_success:
                if success:
                    success_notification = ui.notification(
                        'success', type='positive', position='center',
                        timeout=self.msg_display_time)
                else:
                    success_notification = ui.notification(
                        'failure', type='negative', position='center',
                        timeout=self.msg_display_time)

            self.set_user_data(
                start_notification=start_notification,
                success_notification=success_notification)

@dataclasses.dataclass
class Block:
    stages: List[Stage]
    randomize: bool = False
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


def prepare_blocks(blocks: List[Block]) -> List[Stage]:
    """This function assigns the block metadata to each stage.
    It also flattens all blocks into a single list of stages.
    """
    # assign block description to each stage description
    for block_idx, block in enumerate(blocks):
        for stage in block.stages:
            block.metadata.update(idx=block_idx)
            stage.metadata['block_metadata'] = block.metadata

    # flatten all blocks
    return [stage for block in blocks for stage in block.stages]

def generate_stage_order(blocks: List[Block], block_order: List[int], rng_key: jnp.ndarray) -> List[int]:
    """This function generates the order in which the stages should be displayed.
    It takes the blocks and the block order as input and returns the stage order.

    It also randomizes the order of the stages within each block if the block's randomize flag is True.
    """
    # Assign unique indices to each stage in each block
    block_indices = {}
    current_index = 0
    for block_idx, block in enumerate(blocks):
        block_indices[block_idx] = list(range(current_index, current_index + len(block.stages)))
        current_index += len(block.stages)

    # Generate the final stage order based on block_order
    stage_order = []
    for block_idx in block_order:
        block = blocks[block_idx]
        block_stage_indices = block_indices[block_idx]

        if block.randomize:
            eval_indices = [i for i, stage in enumerate(block.stages)
                            if isinstance(stage, EnvStage) and stage.metadata.get('eval', False)]
            non_eval_indices = [i for i in range(len(block.stages)) if i not in eval_indices]

            if eval_indices:
                rng_key, subkey = jax.random.split(rng_key)
                eval_indices = jax.random.permutation(subkey, jnp.array(eval_indices)).tolist()

            # Combine non-eval indices (in original order) with randomized eval indices
            randomized_indices = non_eval_indices + eval_indices
            block_stage_indices = [block_stage_indices[i] for i in randomized_indices]

        stage_order.extend(block_stage_indices)

    return stage_order

