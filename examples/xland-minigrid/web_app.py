import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import time
import traceback
from datetime import datetime
from importlib.util import find_spec
import shutil
from fastapi import APIRouter

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages

DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}

# Add module loading management
experiment = None
craftax_loaded = asyncio.Event()
load_start_time = None
load_error = None

#####################################
# Setup logger
#####################################
setup_logging(
  DATA_DIR,
  # each user has a unique seed
  # can use this to identify users
  nicegui_storage_user_key="seed",
)
logger = get_logger("main")


#####################################
# Helper functions
#####################################
# This is used to ensure that each user has a unique lock
def get_user_lock():
  """A function that returns a lock for the current user using their unique seed"""
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def experiment_not_finished():
  """Check if the experiment is not finished"""
  global experiment
  async with get_user_lock():
    not_finished = not app.storage.user.get("experiment_finished", False)
    not_finished &= app.storage.user["stage_idx"] < len(experiment.all_stages)
  return not_finished


async def global_handle_key_press(e, container):
  """Define global key press handler

  We can get stage-specific key handling by using this and having this function
  call the stage-specific key handler. When the experiment begins, we'll register
  a key listener to call this function
  """
  global experiment

  if not craftax_loaded.is_set():
    logger.info("craftax not loaded")
    return

  stage_idx = app.storage.user["stage_idx"]
  if app.storage.user["stage_idx"] >= len(experiment.all_stages):
    return

  stage = experiment.all_stages[stage_idx]
  if stage.get_user_data("finished", False):
    return

  await stage.handle_key_press(e, container)
  local_handle_key_press = stage.get_user_data("local_handle_key_press")
  if local_handle_key_press is not None:
    await local_handle_key_press()


def restore_texture_cache_if_needed():
  """Restore texture cache files from local cache if they don't exist in the package directory."""
  # Get paths for texture cache files
  original_constants_directory = os.path.join(
    os.path.dirname(find_spec("craftax.craftax.constants").origin), "assets"
  )
  TEXTURE_CACHE_FILE = os.path.join(original_constants_directory, "texture_cache.pbz2")

  # Local cache paths
  cache_dir = "craftax_cache"
  source_cache = os.path.join(cache_dir, "texture_cache.pbz2")

  # Create the destination directories if they don't exist
  os.makedirs(os.path.dirname(TEXTURE_CACHE_FILE), exist_ok=True)

  # Copy texture cache files if needed
  if not os.path.exists(TEXTURE_CACHE_FILE) and os.path.exists(source_cache):
    logger.info(f"Restoring texture cache from {source_cache} to {TEXTURE_CACHE_FILE}")
    shutil.copy2(source_cache, TEXTURE_CACHE_FILE)
    logger.info("Regular cache file restored successfully!")
  else:
    logger.info(f"{TEXTURE_CACHE_FILE} already exists.")


async def load_craftax_module():
  global experiment, load_start_time, load_error
  load_start_time = datetime.now()
  loop = asyncio.get_event_loop()
  logger.info("Starting craftax module load attempt")

  # Restore texture cache if needed
  restore_texture_cache_if_needed()

  try:
    logger.info("Attempting to import craftax_experiment_structure")

    def import_with_logging():
      try:
        import experiment_structure

        logger.info("Import successful")
        return experiment_structure
      except Exception as e:
        error_msg = f"Import failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

    experiment = await loop.run_in_executor(None, import_with_logging)

    logger.info("Craftax module loaded successfully")
    load_duration = (datetime.now() - load_start_time).total_seconds()
    logger.info(f"Total load time: {load_duration} seconds")

  except Exception as e:
    load_error = str(e)
    error_msg = f"Failed to load craftax module: {str(e)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    raise
  finally:
    craftax_loaded.set()


#####################################
# Setup database for storing experiment data
#####################################
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    # this will use nicewebrl.stages.StageStateModel
    modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


@app.on_startup
async def startup():
  asyncio.create_task(load_craftax_module())
  await init_db()


app.on_shutdown(close_db)


########################
# Run experiment
########################
async def start_experiment(container):
  # --------------------------------
  # initialize
  # place things like consent form here or getting demographic information here
  # --------------------------------
  global experiment

  if not (app.storage.user.get("experiment_started", False)):
    # NOTE: Here is where you can add a consent form or collect demographic information
    app.storage.user["experiment_started"] = True

  # register global key press handler
  ui.on("key_pressed", lambda e: global_handle_key_press(e, container))

  logger.info("Starting experiment")
  while True and await experiment_not_finished():
    # get current stage
    stage_idx = app.storage.user["stage_idx"]
    stage = experiment.all_stages[stage_idx]

    logger.info("=" * 30)
    logger.info(f"Began stage '{stage.name}'")
    # activate stage
    await run_stage(stage, container)
    logger.info(f"Finished stage '{stage.name}'")

    # wait for any saves to finish before updating stage
    # very important, otherwise may lose data
    if isinstance(stage, stages.EnvStage):
      await stage.finish_saving_user_data()
      logger.info(f"Saved data for stage '{stage.name}'")

    # update stage index
    async with get_user_lock():
      app.storage.user["stage_idx"] = stage_idx + 1

    # check if we've finished all stages
    if app.storage.user["stage_idx"] >= len(experiment.all_stages):
      break

  await finish_experiment(container)


async def finish_experiment(container):
  # NOTE: you can do things like saving data here
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")


async def run_stage(stage, container):
  """Runs and Environment Stage


  1. create handlers. We provide handlers for key and button presses as an example.
  2. activate the stage
  3. create button if needed

  """
  #########
  # Create an event to signal when the stage is over
  #########
  stage_over_event = asyncio.Event()

  #########
  # create functions for handling key and button presses
  #########
  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via key press")
        # Signal that the stage is over
        stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  async def handle_button_press():
    # check if stage is already finished, if so, return
    if stage.get_user_data("finished", False):
      return

    # handle button press
    await stage.handle_button_press(container)

    # check if stage is finished, if so, signal that the stage is over
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        # Signal that the stage is over
        logger.info(f"Finished {stage.name} via button press")
        stage_over_event.set()

  #############################################
  # Activate new stage
  #############################################
  with container.style("align-items: center;"):
    await stage.activate(container)

  if stage.get_user_data("finished", False):
    # over as soon as stage activation was complete
    logger.info(f"Finished {stage.name} immediately after activation")
    stage_over_event.set()

  if stage.next_button:
    with container:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)
      await handle_button_press()

  await stage_over_event.wait()


#####################################
# Root page
#####################################


async def check_if_over(container, episode_limit=60):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    # define custom behavior on time-out
    pass


# Add status endpoint for loading screen
router = APIRouter()


@router.get("/status")
async def get_status():
  """Enhanced status endpoint with detailed loading information"""
  global load_start_time, load_error

  current_time = datetime.now()
  load_duration = (
    None
    if load_start_time is None
    else (current_time - load_start_time).total_seconds()
  )

  return {
    "loaded": craftax_loaded.is_set(),
    "load_duration": load_duration,
    "load_error": load_error,
    "load_start_time": load_start_time.isoformat() if load_start_time else None,
  }


app.include_router(router)


# helper to show loading screen
def show_loading_screen(craftax_loaded: asyncio.Event, load_error: str | None = None):
  """Displays a loading UI while waiting for the Craftax module to load."""

  # Inject JS to ping the server periodically
  with open(nicewebrl.basic_javascript_file()) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  # If the module isn't loaded, show loading UI
  if not craftax_loaded.is_set():
    with ui.card().classes("fixed-center") as card:
      card.style("width: 80vw; max-height: 90vh;")

      ui.label("Loading experiment... This will take up to 5 minutes.").classes(
        "text-h4"
      )
      ui.label("Please don't close or refresh the page")

      elapsed_label = ui.label("Time elapsed: 0 seconds")
      status_label = ui.label("Current status: Initializing...")
      error_label = ui.label().classes("text-red")

      start_time = time.time()

      async def update_loading_info():
        seconds = int(time.time() - start_time)
        elapsed_label.text = f"Time elapsed: {seconds} seconds"

        if load_error:
          error_label.text = f"Error: {load_error}"
          status_label.text = "Status: Failed to load"

        if seconds % 10 == 0:
          ui.run_javascript(f"console.log('loading for {seconds} seconds')")
          logger.info(f"Still loading after {seconds} seconds")

        return not craftax_loaded.is_set()

      # Periodic update every 1 second
      ui.timer(1.0, update_loading_info)

      # Add client-side status polling and error logging
      ui.add_body_html("""
                <script>
                let lastPingTime = Date.now();
                
                async function checkStatus() {
                    try {
                        const response = await fetch('/status');
                        const data = await response.json();

                        console.log('Status check:', data);

                        if (data.loaded) {
                            console.log('Module loaded, reloading page');
                            window.location.reload();
                        } else if (data.load_error) {
                            console.error('Loading error:', data.load_error);
                        }

                        const currentTime = Date.now();
                        const timeSinceLastPing = currentTime - lastPingTime;
                        console.log('Time since last ping:', timeSinceLastPing, 'ms');
                        lastPingTime = currentTime;

                        if (!data.loaded) {
                            setTimeout(checkStatus, 1000);
                        }
                    } catch (error) {
                        console.error('Status check failed:', error);
                        setTimeout(checkStatus, 1000);
                    }
                }

                checkStatus();

                window.onerror = function(msg, url, line) {
                    console.error('JavaScript error:', msg, 'at', url, 'line', line);
                    return false;
                };
                </script>
            """)


@ui.page("/")
async def index(request: Request):
  # collect user information from URL (e.g. from MTurk)
  # can save this
  user_info = dict(
    worker_id=request.query_params.get("workerId", None),
    hit_id=request.query_params.get("hitId", None),
    assignment_id=request.query_params.get("assignmentId", None),
  )

  app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
  nicewebrl.initialize_user()

  # set up callback to log all pings
  def print_ping(e):
    logger.info(str(e.args))

  ui.on("ping", print_ping)

  ################
  # Show loading screen
  ################
  if not craftax_loaded.is_set():
    show_loading_screen(craftax_loaded, load_error)
    return

  ################
  # Start experiment
  ################
  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  card = (
    ui.card(align_items=["center"])
    .classes("fixed-center")
    .style(
      "max-width: 90vw;"  # Set the max width of the card
      "max-height: 90vh;"  # Ensure the max height is 90% of the viewport height
      "overflow: auto;"  # Allow scrolling inside the card if content overflows
      "display: flex;"  # Use flexbox for centering
      "flex-direction: column;"  # Stack content vertically
      "justify-content: flex-start;"
      "align-items: center;"
    )
  )
  with card:
    episode_limit = 200
    with ui.column().style("align-items: center;") as container:
      # check every minute if over
      ui.timer(
        interval=1,
        callback=lambda: check_if_over(
          episode_limit=episode_limit, container=container
        ),
      )
      await start_experiment(container)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  host="0.0.0.0",
  port=8080,
  # reload='FLY_ALLOC_ID' not in os.environ,
  reload=True,
  title="XLand-MiniGrid Web App",
)
