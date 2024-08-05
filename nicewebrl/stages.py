from typing import Any, Callable, Dict

from flax import struct
import jax
import jax.numpy as jnp

from nicegui import ui

from nicewebrl.nicejax import new_rng, base64_nparray

def make_image_html(src):
    html = '<div id = "stateImageContainer" >'
    html += f'<img id = "stateImage" src="{src}">'
    html += '</div >'
    return html

@struct.dataclass
class StageState:
    finished: bool = False

@struct.dataclass
class EnvStageState(StageState):
    env_timestep: struct.PyTreeNode = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

@struct.dataclass
class Stage:
    name: str = 'stage'
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

    def restart_env(self, container: ui.element):
        #stage_state = self.new_state(timestep)
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


    def load(self, container: ui.element):
        return self.restart_env(container)
