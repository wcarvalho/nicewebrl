import aiofiles
import msgpack
import os.path
import asyncio
from asyncio import Lock
from datetime import datetime, timedelta
from typing import Callable, Awaitable
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator

# from gcs import save_to_gcs_with_retries
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages


import experiment

DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

DEBUG = int(os.environ.get('DEBUG', 0))
DEBUG_SEED = int(os.environ.get('SEED', 0))
NAME = os.environ.get('NAME', 'exp')
DATABASE_FILE = f'{DATABASE_FILE}_name={NAME}_debug={DEBUG}'

os.makedirs(DATA_DIR, exist_ok=True)

_user_locks = {}


#####################################
# Setup logger
#####################################
def log_filename_fn(log_dir, user_id):
  return os.path.join(log_dir, f'log_{user_id}.log')


setup_logging(
    DATA_DIR,
    # each user has a unique seed
    # can use this to identify users
    log_filename_fn=log_filename_fn,
    nicegui_storage_user_key='seed')
logger = get_logger('main')

#####################################
# Helper functions
#####################################
def stage_name(stage):
  return stage.name

# This is used to ensure that each user has a unique lock
def get_user_lock():
    """A function that returns a lock for the current user using their unique seed"""
    user_seed = app.storage.user['seed']
    if user_seed not in _user_locks:
        _user_locks[user_seed] = Lock()
    return _user_locks[user_seed]


async def experiment_not_finished():
  """Check if the experiment is not finished"""
  async with get_user_lock():
    not_finished = not app.storage.user.get('experiment_finished', False)
    not_finished &= app.storage.user['stage_idx'] < len(experiment.all_stages)
  return not_finished


def blob_user_filename():
  """filename structure for user data in GCS (cloud)"""
  seed = app.storage.user['seed']
  worker = app.storage.user.get('worker_id', None)
  if worker is not None:
    return f'user={seed}_worker={worker}_name={NAME}_debug={DEBUG}'
  else:
    return f'user={seed}_name={NAME}_debug={DEBUG}'


async def global_handle_key_press(e, container):
    """Define global key press handler

    We can get stage-specific key handling by using this and having this function
    call the stage-specific key handler. When the experiment begins, we'll register
    a key listener to call this function
    """

    stage_idx = app.storage.user['stage_idx']
    if app.storage.user['stage_idx'] >= len(experiment.all_stages):
        return

    stage = experiment.all_stages[stage_idx]
    if stage.get_user_data('finished', False):
      return

    await stage.handle_key_press(e, container)
    local_handle_key_press = stage.get_user_data('local_handle_key_press')
    if local_handle_key_press is not None:
      await local_handle_key_press()


async def save_data(final_save=True, feedback=None, **kwargs):
    user_data_file = experiment.get_user_save_file_fn()

    if final_save:
      # --------------------------------
      # save user data to final line of file
      # --------------------------------
      user_storage = nicewebrl.make_serializable(
          dict(app.storage.user))
      last_line = dict(
          finished=True,
          feedback=feedback,
          user_storage=user_storage,
          **kwargs,
      )
      async with aiofiles.open(user_data_file, 'ab') as f:  # Changed to binary mode
          # Use msgpack to serialize the data
          packed_data = msgpack.packb(last_line)
          await f.write(packed_data)
          await f.write(b'\n')  # Add newline in binary mode

    if not DEBUG:
        files_to_save = [
            (user_data_file, f'data/{blob_user_filename()}.json'),
            (log_filename_fn(DATA_DIR, app.storage.user.get('user_id')),
              f'logs/{blob_user_filename()}.log')
        ]
        # await save_to_gcs_with_retries(
        #    files_to_save,
        #    max_retries=5 if final_save else 1,
        #    )

#####################################
# Setup database for storing experiment data
#####################################
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


async def init_db() -> None:
    await Tortoise.init(
        db_url=f'sqlite://{DATA_DIR}/{DATABASE_FILE}',
        # this will look in models.py,
        # models.py uses defaults from nicewebrl
        modules={"models": ["nicewebrl.stages"]})
    await Tortoise.generate_schemas()


async def close_db() -> None:
    await Tortoise.close_connections()

app.on_startup(init_db)
app.on_shutdown(close_db)

#####################################
# Consent Form and demographic info
#####################################


async def make_consent_form(container):
  consent_given = asyncio.Event()
  with container:
    ui.markdown('## Consent Form')
    with open('misc/consent.md', 'r') as consent_file:
        consent_text = consent_file.read()
    ui.markdown(consent_text)
    ui.checkbox('I agree to participate.',
                on_change=lambda: consent_given.set())

  await consent_given.wait()


async def collect_demographic_info(container):
    # Create a markdown title for the section
    nicewebrl.clear_element(container)
    with container:
      ui.markdown('## Demographic Info')
      ui.markdown('Please fill out the following information.')

      with ui.column():
        with ui.column():
          ui.label('Biological Sex')
          sex_input = ui.radio(
              ['Male', 'Female'], value='Male').props('inline')

        # Collect age with a textbox input
        age_input = ui.input('Age')

      # Button to submit and store the data
      async def submit():
          age = age_input.value
          sex = sex_input.value

          # Validation for age input
          if not age.isdigit() or not (0 < int(age) < 100):
              ui.notify(
                  "Please enter a valid age between 1 and 99.", type="warning")
              return
          app.storage.user['age'] = int(age)
          app.storage.user['sex'] = sex
          logger.info(f"age: {int(age)}, sex: {sex}")

      button = ui.button('Submit', on_click=submit)
      await button.clicked()

########################
# Run experiment
########################

async def start_experiment(meta_container, stage_container, button_container):

  #========================================
  # Consent form and demographic info
  #========================================
  if not (app.storage.user.get('experiment_started', False) or DEBUG):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user['experiment_started'] = True

  # ========================================
  # Force fullscreen
  # ========================================
  if DEBUG == 0:
    ui.run_javascript(
        'document.documentElement.requestFullscreen()')
    ui.run_javascript('window.require_fullscreen = true')
  else:
    ui.run_javascript('window.require_fullscreen = false')

  # ========================================
  # Register global key press handler
  # ========================================
  ui.on('key_pressed', lambda e: global_handle_key_press(e, meta_container))

  # ========================================
  # Run experiment
  # ========================================
  logger.info("Starting experiment")
  while True and await experiment_not_finished():
      # get current stage
      stage_idx = app.storage.user['stage_idx']
      stage = experiment.all_stages[stage_idx]

      logger.info("="*30)
      logger.info(f"Began stage '{stage.name}'")
      # activate stage
      await run_stage(stage, stage_container, button_container)
      logger.info(f"Finished stage '{stage.name}'")
      ui.notify("Loading next page...", type='info', position='top')

      # wait for any saves to finish before updating stage
      # very important, otherwise may lose data
      if isinstance(stage, stages.EnvStage):
        await stage.finish_saving_user_data()
        logger.info(f"Saved data for stage '{stage.name}'")

      # update stage index
      async with get_user_lock():
        app.storage.user['stage_idx'] = stage_idx + 1

      # check if we've finished all stages
      if app.storage.user['stage_idx'] >= len(experiment.all_stages):
          break

  await finish_experiment(meta_container, stage_container, button_container)


async def finish_experiment(meta_container, stage_container, button_container):
    nicewebrl.clear_element(meta_container)
    nicewebrl.clear_element(stage_container)
    nicewebrl.clear_element(button_container)
    logger.info("Finishing experiment")
    experiment_finished = app.storage.user.get('experiment_finished', False)

    if experiment_finished and not DEBUG:
      # in case called multiple times
      return

    #########################
    # Save data
    #########################
    async def submit(feedback):
      app.storage.user['experiment_finished'] = True
      with meta_container:
        nicewebrl.clear_element(meta_container)
        ui.markdown(f"## Saving data. Please wait")
        ui.markdown(
            "**Once the data is uploaded, this app will automatically move to the next screen**")

      # when over, delete user data.
      await save_data(final_save=True, feedback=feedback)
      app.storage.user['data_saved'] = True

    app.storage.user['data_saved'] = app.storage.user.get(
        'data_saved', False)
    if not app.storage.user['data_saved']:
      with meta_container:
        nicewebrl.clear_element(meta_container)
        ui.markdown("Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment.")
        text = ui.textarea().style('width: 80%;')  # Set width to 80% of the container
        button = ui.button("Submit")
        await button.clicked()
        await submit(text.value)

    #########################
    # Final screen
    #########################
    with meta_container:
        nicewebrl.clear_element(meta_container)
        ui.markdown("# Experiment over")
        ui.markdown("## Data saved")
        ui.markdown(
            "### Please record the following code which you will need to provide for compensation")
        ui.markdown(
            f'### socialrl.cook')
        ui.markdown("#### You may close the browser")


async def run_stage(stage, stage_container, button_container):
  #########
  # create functions for handling key and button presses
  # Create an event to signal when the stage is over
  #########
  stage_over_event = asyncio.Event()

  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data('finished', False):
          # Signal that the stage is over
          logger.info(f"Finished {stage_name(stage)} via key press")
          ui.notify("Loading next page...", type='info', position='top')
          stage_over_event.set()

  async def handle_button_press():
    if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
      ui.notify('Please enter fullscreen mode to continue experiment',
                type='negative')
      logger.info("Button press but not fullscreen")
      return
    if stage.get_user_data('finished', False):
       return
    # nicewebrl.clear_element(button_container)
    await stage.handle_button_press(stage_container)
    async with get_user_lock():
      if stage.get_user_data('finished', False):
          # Signal that the stage is over
          logger.info(f"Finished {stage_name(stage)} via button press")
          ui.notify("Loading next page...", type='info', position='top')
          stage_over_event.set()

  #############################################
  # Activate new stage
  #############################################
  with stage_container.style('align-items: center;'):
    await stage.activate(stage_container)

  if stage.get_user_data('finished', False):
    # over as soon as stage activation was complete
    logger.info(f"Finished {stage_name(stage)} immediately after activation")
    ui.notify("Loading next page...", type='info', position='top')
    stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  with button_container.style('align-items: center;'):
      nicewebrl.clear_element(button_container)
      ####################
      # Button to go to next page
      ####################
      checking_fullscreen = DEBUG == 0
      next_button_container = ui.row()

      async def create_button_and_wait():
          with next_button_container:
            nicewebrl.clear_element(next_button_container)
            button = ui.button('Next page').bind_visibility_from(
                stage, 'next_button')
            await wait_for_button_or_keypress(button)
            logger.info("Button or key pressed")
            await handle_button_press()
      if stage.next_button:
        if checking_fullscreen:
            await create_button_and_wait()
            while not await nicewebrl.utils.check_fullscreen():
              if await stage_over_event.wait():
                 break
              logger.info("Waiting for fullscreen")
              await asyncio.sleep(0.1)
              await create_button_and_wait()
        else:
          await create_button_and_wait()

  await stage_over_event.wait()
  # nicewebrl.clear_element(button_container)
  nicewebrl.clear_element(button_container)


#####################################
# Root page
#####################################

async def check_if_over(container, episode_limit=60):
   minutes_passed = nicewebrl.get_user_session_minutes()
   minutes_passed = app.storage.user['session_duration']
   if minutes_passed > episode_limit:
      # define custom behavior on time-out
      pass

def initalize_user(request: Request):
  nicewebrl.initialize_user(seed=DEBUG_SEED)
  app.storage.user['worker_id'] = request.query_params.get('workerId', None)
  app.storage.user['hit_id'] = request.query_params.get('hitId', None)
  app.storage.user['assignment_id'] = request.query_params.get('assignmentId', None)

  app.storage.user['user_id'] = app.storage.user['worker_id'] or app.storage.user['seed']
  app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)


@ui.page('/')
async def index(request: Request):
    initalize_user(request)

    ui.run_javascript(f'window.debug = {DEBUG}')
    # set up callback to log all pings
    def print_ping(e):
      logger.info(str(e.args))
    ui.on('ping', print_ping)

    ################
    # Start experiment
    ################
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    card = ui.card(align_items=['center']).classes('fixed-center').style(
        'max-width: 90vw;'  # Set the max width of the card
        'max-height: 90vh;'  # Ensure the max height is 90% of the viewport height
        'overflow: auto;'  # Allow scrolling inside the card if content overflows
        'display: flex;'  # Use flexbox for centering
        'flex-direction: column;'  # Stack content vertically
        'justify-content: flex-start;'
        'align-items: center;'
    )
    with card:
      episode_limit = 200
      meta_container = ui.column()
      with meta_container.style('align-items: center;'):
        stage_container = ui.column()
        button_container = ui.column()
        ui.timer(
            interval=1,
            callback=lambda: check_if_over(
                episode_limit=episode_limit,
                meta_container=meta_container,
                stage_container=stage_container,
                button_container=button_container))
        footer_container = ui.row()
        footer(footer_container)
      with meta_container.style('align-items: center;'):
        await start_experiment(meta_container, stage_container, button_container)

async def check_if_over(*args, episode_limit=60, ** kwargs):
   """If past time limit, finish experiment"""
   minutes_passed = nicewebrl.get_user_session_minutes()
   minutes_passed = app.storage.user['session_duration']
   if minutes_passed > episode_limit:
      logger.info(f"experiment timed out after {minutes_passed} minutes")
      app.storage.user['stage_idx'] = len(experiment.all_stages)
      await finish_experiment(*args, **kwargs)

def footer(footer_container):
  """Add user information and progress bar to the footer"""
  with footer_container:
    with ui.row():
        ui.label().bind_text_from(
            app.storage.user, 'seed',
            lambda v: f"user id: {v}.")
        ui.label()
        ui.label().bind_text_from(
            app.storage.user, 'stage_idx',
            lambda v: f"stage: {int(v) + 1}/{len(experiment.all_stages)}.")
        ui.label()
        ui.label().bind_text_from(
            app.storage.user, 'session_duration',
            lambda v: f"minutes passed: {int(v)}.")

    stage_progress = lambda: float(f"{(app.storage.user['stage_idx']+1)/len(experiment.all_stages):.2f}")

    ui.linear_progress(
        value=stage_progress()).bind_value_from(app.storage.user, 'stage_progress')

    ui.button(
        'Toggle fullscreen', icon='fullscreen',
        on_click=nicewebrl.utils.toggle_fullscreen).props('flat')

ui.run(
    storage_secret='private key to secure the browser session cookie',
     reload='FLY_ALLOC_ID' not in os.environ,
    #reload=False,
    title='Crafter Web App',
)