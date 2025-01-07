from typing import List, Any, Callable, Dict, Optional, Union

import asyncio
from asyncio import Lock
import aiofiles
import copy
import dataclasses

from flax import struct
from flax import serialization
import jax

import nicewebrl
from nicegui import app, ui
from nicewebrl.nicejax import new_rng, base64_npimage, make_serializable, TimeStep
from nicewebrl.utils import retry_with_exponential_backoff
import msgpack

@dataclasses.dataclass
class MultiHumanEnvStage(nicewebrl.Stage):
    """A stage class for handling interactive environment episodes.

    This class manages the interaction between a user and an environment, handling
    state transitions, user inputs, and data collection.

    Args:
        instruction (str): Text instructions shown to the user for this stage.
        max_episodes (Optional[int]): Maximum number of episodes allowed before stage completion.
        min_success (Optional[int]): Minimum number of successful episodes required to complete stage.
        web_env (Any): The environment instance that handles state transitions and interactions.
        env_params (struct.PyTreeNode): Parameters for the environment.
        render_fn (Callable): Function to render the environment state as an image.
        reset_display_fn (Callable): Function called to reset the display between episodes.
        vmap_render_fn (Callable): Vectorized version of render_fn for batch processing.
        evaluate_success_fn (Callable): Function that takes a timestep and returns 1 for success, 0 for failure.
        check_finished (Callable): Additional function to check if stage should end (beyond max_episodes/min_success).
        custom_data_fn (Callable): Optional function to extract additional data from timesteps for logging.
        state_cls (EnvStageState): Class used to store the stage's state information.
        action_to_key (Dict[int, str]): Mapping from action indices to keyboard keys.
        action_to_name (Dict[int, str]): Optional mapping from action indices to human-readable names.
        next_button (bool): Whether to show a "next" button (default False).
        notify_success (bool): Whether to show success/failure notifications.
        msg_display_time (int): How long to display notification messages (in milliseconds).
        end_on_final_timestep (bool): Whether to end the stage on the final timestep.
        user_save_file_fn (Callable[[], str]): Function that returns the path to save user data.
        verbosity (int): Level of logging verbosity (0 for minimal, higher for more).
    """

    instruction: str = 'instruction'
    min_success: Optional[int] = 1
    max_episodes: Optional[int] = 10
    web_env: nicejax.JaxWebEnv = None
    env_params: struct.PyTreeNode = None
    render_fn: RenderFn = None
    reset_display_fn: Optional[DisplayFn] = None
    vmap_render_fn: Optional[Callable] = None
    evaluate_success_fn: TimestepCallFn = None
    check_finished: Optional[TimestepCallFn] = None
    custom_data_fn: Optional[Callable] = None
    state_cls: Optional[EnvStageState] = None
    action_keys: Optional[Dict[int, str]] = None
    action_to_name: Optional[List[str]] = None
    next_button: bool = False
    notify_success: bool = True
    msg_display_time: int = None
    user_save_file_fn: Optional[Callable[[], str]] = None
    verbosity: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.vmap_render_fn is None:
            self.vmap_render_fn = self.web_env.precompile_vmap_render_fn(
                self.render_fn, self.env_params)

        self.key_to_action = {k: a for a, k in enumerate(self.action_keys)}
        if self.action_to_name is None:
            self.action_to_name = dict()
        else:
            self.action_to_name = {k: v for k,
                                   v in enumerate(self.action_to_name)}

        if self.user_save_file_fn is None:
            self.user_save_file_fn = lambda: f'data/user={app.storage.user.get("seed")}.json'

        if self.check_finished is None:
            self.check_finished = lambda timestep: False

        if self.state_cls is None:
            self.state_cls = EnvStageState

        self._user_queues = {}  # new: dictionary to store per-user queues

    def get_user_queue(self):
        """Get queue for current user, creating if needed"""
        user_seed = app.storage.user['seed']
        if user_seed not in self._user_queues:
            self._user_queues[user_seed] = asyncio.Queue()
        return self._user_queues[user_seed]

    async def finish_saving_user_data(self):
        await self.get_user_queue().join()

    async def _process_save_queue(self):
        """Process all items currently in the queue for current user"""
        queue = self.get_user_queue()
        while not queue.empty():
            args, timestep, user_stats = await queue.get()
            await self.save_experiment_data(
                args,
                timestep=timestep,
                user_stats=user_stats,
            )
            queue.task_done()

    async def step_and_send_timestep(
            self,
            container,
            timestep,
            update_display: bool = True):
        #############################
        # get next images and store them client-side
        # setup code to display next state
        #############################
        rng = new_rng()
        next_timesteps = self.web_env.next_steps(
            rng, timestep, self.env_params)
        next_images = self.vmap_render_fn(next_timesteps)

        next_images = {
            self.action_keys[idx]: base64_npimage(image) for idx, image in enumerate(next_images)}

        js_code = f"window.next_states = {next_images};"

        ui.run_javascript(js_code)

        await self.set_user_data(next_timesteps=next_timesteps)
        #############################
        # display image
        #############################
        if update_display:
            await self.display_fn(
                stage=self,
                container=container,
                timestep=timestep,
            )

            attempt = 0
            while True:
                attempt += 1
                try:
                    # Set the timestamp in the browser
                    await ui.run_javascript(
                        "window.imageSeenTime = new Date();",
                        timeout=2
                    )
                    # If successful, we can return immediately
                    return
                except Exception as e:
                    if attempt % 10 == 0:  # Log every 10 attempts
                        logger.warning(
                            f"{self.name}: Error getting imageSeenTime (attempt {attempt}): {e}")
                    await asyncio.sleep(0.1)  # Short delay between attempts
                    if attempt > 100:
                        ui.notify(f"Please refresh the page", type='negative')
                        return

        else:
            ui.run_javascript(
                "window.imageSeenTime = window.next_imageSeenTime;", timeout=10)

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

    async def reset_stage(self) -> EnvStageState:
        rng = new_rng()

        # NEW EPISODE
        timestep = self.web_env.reset(rng, self.env_params)
        return self.state_cls(timestep=timestep)

    async def activate(self, container: ui.element):
        """
        
        First reset stage and get a new stage state.
        Then try to load stage state from memory using the stage state to get the right types.
        If no stage state is found, continue with the new stage state.
        """

        async with self.get_user_lock():
            if self.verbosity:
                logger.info("="*30)
            if self.verbosity:
                logger.info(self.metadata)

            # reset stage
            rng = new_rng()
            timestep = self.web_env.reset(rng, self.env_params)
            new_stage_state = self.state_cls(timestep=timestep)

            # (potentially) load stage state from memory
            loaded_stage_state = await get_latest_stage_state(
                example=new_stage_state)

            if loaded_stage_state is None:
                logger.info("No stage state found, starting new stage")
                # await self.start_stage(container, new_stage_state)
                await self.set_user_data(stage_state=new_stage_state)
                asyncio.create_task(save_stage_state(new_stage_state))

                # DISPLAY NEW EPISODE
                await self.wait_for_start(container, new_stage_state.timestep)
                await self.step_and_send_timestep(
                    container, new_stage_state.timestep)

            else:
                logger.info("Loading stage state from memory")
                # await self.load_stage(container, loaded_stage_state)
                await self.set_user_data(stage_state=loaded_stage_state)
                await self.step_and_send_timestep(container, loaded_stage_state.timestep)

            await self.set_user_data(started=True)
            ui.run_javascript("window.accept_keys = true;")

    def user_stats(self):
        stage_state = self.get_user_data('stage_state')
        if stage_state is None:
            return dict()
        return dict(
            nsteps=int(stage_state.nsteps),
            nepisodes=int(stage_state.nepisodes),
            nsuccesses=int(stage_state.nsuccesses),
        )

    async def save_experiment_data(self, args, timestep, user_stats):
        key = args['key']
        keydownTime = args.get('keydownTime')
        imageSeenTime = args.get('imageSeenTime')
        action_idx = self.key_to_action.get(key, -1)
        action_name = self.action_to_name.get(action_idx, key)

        timestep_data = {}
        if self.custom_data_fn is not None:
            timestep_data = self.custom_data_fn(timestep)
            timestep_data = jax.tree_map(make_serializable, timestep_data)

        serialized_timestep = serialization.to_bytes(timestep)

        step_metadata = copy.deepcopy(self.metadata)
        step_metadata.update(type='EnvStage', **user_stats)

        user_data = dict(
            user_id=app.storage.user['seed'],
            age=app.storage.user.get('age'),
            sex=app.storage.user.get('sex'),
        )

        save_data = dict(
            stage_idx=app.storage.user.get('stage_idx'),
            session_id=app.storage.browser['id'],
            data=dict(
                image_seen_time=imageSeenTime,
                action_taken_time=keydownTime,
                computer_interaction=key,
                action_name=action_name,
                action_idx=action_idx,
                timelimit=self.duration,
                timestep=serialized_timestep,
                **timestep_data,
            ),
            user_data=user_data,
            metadata=step_metadata,
            name=self.name,
            body=self.body,
        )

        # Use aiofiles for async file I/O
        save_file = self.user_save_file_fn()
        async with aiofiles.open(save_file, 'ab') as f:  # Note: open in binary mode
            # Use msgpack to serialize the data, including bytes
            packed_data = msgpack.packb(save_data)
            await f.write(packed_data)
            await f.write(b'\n')  # Add newline in binary mode
            name = self.metadata.get('maze', self.name)
            if imageSeenTime is not None and keydownTime is not None:
                stage_state = self.get_user_data('stage_state')
                if self.verbosity:
                    logger.info(f'{name} saved file')
                    logger.info(
                        f'∆t: {time_diff(imageSeenTime, keydownTime)/1000.}')
                    logger.info(f'stage state: {self.user_stats()}')
                    logger.info(f'env step: {stage_state.nsteps}')

            else:
                logger.error(f'{name} saved file')
                logger.error(f'stage state: {self.user_stats()}')
                logger.error(
                    f"imageSeenTime={imageSeenTime}, keydownTime={keydownTime}")
                ui.notification(
                    "Error: Stage unexpectedly ending early",
                    type='negative')
                await self.set_user_data(
                    finished=True,
                    final_save=True)

        await self.set_user_data(saved_data=True)

    @retry_with_exponential_backoff(max_retries=5, base_delay=1, max_delay=10)
    async def finish_stage(self):
        if not self.get_user_data('started', False):
            return
        if self.get_user_data('finished', False):
            return

        # Wait for any pending saves to complete
        await self.get_user_queue().join()

        # save experiment data so far (prior time-step + resultant action)
        # if finished, save synchronously (to avoid race condition) with next stage
        await self.set_user_data(
            finished=True,
            final_save=True)
        logger.info(f"finish_stage {self.name}. stats: {self.user_stats()}")
        imageSeenTime = await ui.run_javascript('getImageSeenTime()', timeout=10)

        start_notification = self.pop_user_data('start_notification')
        if start_notification:
            start_notification.dismiss()
        success_notification = self.pop_user_data('success_notification')
        if success_notification:
            success_notification.dismiss()

        stage_state = self.get_user_data('stage_state')
        await self.save_experiment_data(
            args=dict(
                key='timer',
                keydownTime=imageSeenTime,
                imageSeenTime=imageSeenTime,
            ),
            timestep=stage_state.timestep,
            user_stats=self.user_stats(),
        )

    async def handle_key_press(self, event, container):
        # Get or create lock for this specific user
        async with self.get_user_lock():
            await self._handle_key_press(event, container)

    async def _handle_key_press(self, event, container):
        if not self.get_user_data('started', False):
            return
        if self.get_user_data('stage_finished', False):
            # if already did final save, just return
            if self.get_user_data('final_save', False):
                return

            # did not do final save, so do so now
            # want stage to end on keypress so that
            # notifications are visible at final timestep
            await self.finish_stage()
            # and dismiss any present notifications
            start_notification = self.pop_user_data('start_notification')
            if start_notification:
                start_notification.dismiss()
            success_notification = self.pop_user_data('success_notification')
            if success_notification:
                success_notification.dismiss()
            return

        key = event.args['key']
        if self.verbosity:
            logger.info(f'handle_key_press key: {key}')

        # check if valid environment interaction
        if not key in self.key_to_action:
            return

        # asynchonously save experiment data by putting in a save queue
        # save prior timestep + current event information
        user_stats = self.user_stats()
        timestep = self.get_user_data('stage_state').timestep
        await self.get_user_queue().put((event.args, timestep, user_stats))
        asyncio.create_task(self._process_save_queue())

        # use action to select from avaialble next time-steps
        action_idx = self.key_to_action[key]
        next_timesteps = self.get_user_data('next_timesteps')
        timestep = jax.tree_map(lambda t: t[action_idx], next_timesteps)

        episode_reset = timestep.first()
        if episode_reset:
            start_notification = self.pop_user_data('start_notification')
            if start_notification:
                start_notification.dismiss()
            success_notification = self.pop_user_data('success_notification')
            if success_notification:
                success_notification.dismiss()

        success = self.evaluate_success_fn(timestep)

        stage_state = self.get_user_data('stage_state')
        stage_state = stage_state.replace(
            timestep=timestep,
            nsteps=stage_state.nsteps + 1,
            nepisodes=stage_state.nepisodes + timestep.first(),
            nsuccesses=stage_state.nsuccesses + success,
        )

        # asynchronously save stage state
        asyncio.create_task(save_stage_state(stage_state))
        await self.set_user_data(stage_state=stage_state)

        ################
        # Stage over?
        ################
        achieved_min_success = stage_state.nsuccesses >= self.min_success
        achieved_max_episodes = stage_state.nepisodes >= self.max_episodes and timestep.last()
        finished = (achieved_min_success or achieved_max_episodes)
        stage_finished = finished or self.check_finished(timestep)

        ################
        # Display new data?
        ################
        if episode_reset:
            await self.wait_for_start(container, timestep)
        await self.step_and_send_timestep(
            container, timestep,
            # image is normally updated client-side
            # when episode resets, update server-side
            update_display=episode_reset,
        )
        ################
        # Episode over?
        ################
        if timestep.last():
            if self.verbosity:
                logger.info("-"*20)
                logger.info("episode over")
                logger.info("-"*20)
            start_notification = None
            if not stage_finished:
                start_notification = ui.notification(
                    'press any arrow key to start next episode',
                    position='center', type='info', timeout=self.msg_display_time)
            else:
                start_notification = ui.notification(
                    'press any arrow key to continue',
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

            await self.set_user_data(
                start_notification=start_notification,
                success_notification=success_notification)

        await self.set_user_data(stage_finished=stage_finished)

    async def handle_button_press(self, container): pass  # do nothing
