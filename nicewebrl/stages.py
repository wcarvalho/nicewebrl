from typing import Any, Callable, Dict
import base64

from flax import struct
from flax import serialization
import jax
import jax.numpy as jnp

from nicegui import app, ui
from nicewebrl import nicejax
from nicewebrl.nicejax import new_rng, base64_nparray, make_serializable
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


async def get_latest_stage_state(cls: struct.PyTreeNode) -> StageStateModel | None:
    latest = await StageStateModel.filter(
        session_id=app.storage.browser['id'],
        stage_idx=app.storage.user['stage_idx'],
    ).order_by('-id').first()

    if latest is not None:
        latest = nicejax.deserialize_bytes(cls, latest)
    return latest

@struct.dataclass
class StageState:
    finished: bool = False

@struct.dataclass
class EnvStageState(StageState):
    timestep: struct.PyTreeNode = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

@struct.dataclass
class Stage:
    name: str = 'stage'
    state_cls: StageState = StageState

    def load(self, container: ui.element): pass

@struct.dataclass
class EnvStage(Stage):
    instruction: str = 'instruction'
    web_env: Any = None
    action_to_key: Dict[int, str] = None
    env_params: struct.PyTreeNode = None
    reset_env: bool = True
    render_fn: Callable = None
    vmap_render_fn: Callable = None
    task_desc_fn: Callable = None
    state_cls: StageState = None

    def restart_env(self, container: ui.element):
        rng = new_rng()
        #############################
        # get current image and display it
        #############################
        timestep = self.web_env.reset(rng, self.env_params)
        image = self.render_fn(timestep)
        image = base64_nparray(image)
        with container.style('align-items: center;'):
            container.clear()
            ui.markdown(f"# {self.name}")
            ui.markdown(f"### {self.instruction}")
            self.task_desc_fn(timestep)
            ui.html(make_image_html(src=image))
            ui.run_javascript("window.imageSeenTime = new Date();")
            #ui.html(make_image_html(src=''))
            #js = f"document.getElementById('stateImage').src={image};"
            #js += "\nwindow.imageSeenTime = new Date();"
            #print(js)
            ## for accurate times
            #ui.run_javascript(js)


        #############################
        # get next images and store them client-side
        # setup code to display next wind
        #############################
        rng = new_rng()
        next_timesteps = self.web_env.next_steps(
            rng, timestep, self.env_params)
        next_images = self.vmap_render_fn(next_timesteps)
        next_images = {
            self.action_to_key[idx]: base64_nparray(image) for idx, image in enumerate(next_images)}
        js_code = f"window.next_states = {next_images};"
        ui.run_javascript(js_code)

        return timestep

    async def start_stage(self, container: ui.element):
        timestep = self.restart_env(container)
        stage_state = self.state_cls(timestep=timestep)
        stage_state = jax.tree_map(make_serializable, stage_state)
        serialized_data = serialization.to_bytes(stage_state)
        encoded_data = base64.b64encode(serialized_data).decode('ascii')
        #serialized_data_str = serialized_data.decode('utf-8')
        model = StageStateModel(
            session_id=app.storage.browser['id'],
            stage_idx=app.storage.user['stage_idx'],
            data=encoded_data,
        )
        await model.save()
        nicejax.deserialize_bytes(self.state_cls, encoded_data)

    async def load(self,
             container: ui.element,
             state: StageState = None,
             ):
        #if state is None:
        return await self.start_stage(container)
        #else:
        #    import ipdb; ipdb.set_trace()
