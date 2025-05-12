import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import jax
import httpx
import config
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages
import time
from google_cloud_utils import save_to_gcs_with_retries, save_file_to_gcs

from experiment_structure import experiment
import config


DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}

# previous_obs_base64 = None
# current_obs_base64 = None

def convert_state_to_text(state):
  gamestate = state.state

  agent_position = gamestate.agent.position.tolist()
  agent_direction = gamestate.agent.direction
  # convert agent integer direction to string
  if agent_direction == 0:
    agent_direction = "UP"
  elif agent_direction == 1:
    agent_direction = "RIGHT"
  elif agent_direction == 2:
    agent_direction = "DOWN"
  elif agent_direction == 3:
    agent_direction = "LEFT"

  cur_reward = state.reward
  cur_grid = gamestate.grid
  cur_grid = cur_grid.tolist()

  state_text = (
    f"Agent Position: {agent_position}, Agent Direction: {agent_direction}\n"
  )
  state_text += f"Current Reward: {cur_reward}\n"
  state_text += """Each point in the grid is represented as a tuple (object_type, color), where:

  - object_type is an integer from the Tiles class:
      EMPTY = 0
      FLOOR = 1
      WALL = 2
      BALL = 3
      SQUARE = 4
      PYRAMID = 5
      GOAL = 6
      KEY = 7
      DOOR_LOCKED = 8
      DOOR_CLOSED = 9
      DOOR_OPEN = 10
      HEX = 11
      STAR = 12

  - color is an integer from the Colors class:
      EMPTY = 0
      RED = 1
      GREEN = 2
      BLUE = 3
      PURPLE = 4
      YELLOW = 5
      GREY = 6
      BLACK = 7
      ORANGE = 8
      WHITE = 9
      BROWN = 10
      PINK = 11

  Examples:
      (3, 1)  -> red BALL
      (6, 9)  -> white GOAL

      We now give the current grid state:
  """
  for i in range(len(cur_grid)):
    for j in range(len(cur_grid[i])):
      obj_type, color = cur_grid[i][j]
      state_text += f"({obj_type}, {color}) "
    state_text += "\n"
  state_text += "\n"
  return state_text


async def get_gemini_response(message, env_text):
  url = f"{config.GEMINI_API_URL}?key={config.GEMINI_API_KEY}"
  headers = {"Content-Type": "application/json"}

  parts = [
    {
      "text": (
        "You are a helpful assistant for a Minigrid 8x8 Empty reinforcement learning game.\n"
        "These are the keys to control the agent:\n"
        "ArrowUp: Move Forward\n"
        "ArrowRight: Turn Right\n"
        "ArrowLeft: Turn Left\n"
        "p: Pick Up\n"
        "d: Drop\n"
        "t: Toggle\n"
        "The agent can pick up, drop, and toggle the doors.\n"
        "Current environment state:\n"
        f"{env_text}\n"
        "Use this information to understand the user's position and goal.\n"
        "Give short, specific hints to help them progress.\n"
        "Keep responses to 1-2 lines."
      )
    }
  ]
  parts.append({"text": f"Question: {message}"})
  data = {"contents": [{"parts": parts}]}

  async with httpx.AsyncClient() as client:
    res = await client.post(url, headers=headers, json=data)
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]

async def get_claude_response(message, env_text):
  url = config.CLAUDE_API_URL
  headers = {
    "Content-Type": "application/json",
    "x-api-key": config.CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01",
  }

  system_prompt = (
    "You are a helpful assistant for a Gridworld reinforcement learning game.\n"
    "These are the keys to control the agent:\n"
    "ArrowUp: Move Forward\n"
    "ArrowRight: Turn Right\n"
    "ArrowLeft: Turn Left\n"
    "p: Pick Up\n"
    "d: Drop\n"
    "t: Toggle\n"
    "The agent can pick up, drop, and toggle the doors.\n"
    "Use this information to understand the user's position and goal.\n"
    "Give short, specific hints to help them progress.\n"
    "Keep responses to 1-2 lines.\n\n"
  )
  user_prompt = f"Current environment state:\n{env_text}\nQuestion: {message}"

  data = {
    "model": config.CLAUDE_MODEL,
    "max_tokens": 150,
    "system": system_prompt,
    "messages": [{"role": "user", "content": user_prompt}],
  }

  async with httpx.AsyncClient() as client:
    res = await client.post(url, headers=headers, json=data)
    res.raise_for_status()
    return res.json()["content"][0]["text"]

async def get_chatgpt_response(message, env_text):
  url = config.CHATGPT_API_URL
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.CHATGPT_API_KEY}",
  }

  prompt = (
    "You are a helpful assistant for a Minigrid 8x8 Empty reinforcement learning game.\n"
    "These are the keys to control the agent:\n"
    "ArrowUp: Move Forward\n"
    "ArrowRight: Turn Right\n"
    "ArrowLeft: Turn Left\n"
    "p: Pick Up\n"
    "d: Drop\n"
    "t: Toggle\n"
    "The agent can pick up, drop, and toggle the doors.\n"
    "Current environment state:\n"
    f"{env_text}\n"
    "Use this information to understand the user's position and goal.\n"
    "Give short, specific hints to help them progress.\n"
    "Keep responses to 1-2 lines."
  )

  data = {
    "model": config.CHATGPT_MODEL,
    "messages": [
      {"role": "system", "content": prompt},
      {"role": "user", "content": message},
    ],
    "max_tokens": 150,
  }

  async with httpx.AsyncClient() as client:
    res = await client.post(url, headers=headers, json=data)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

async def send_message(chat_input, response_box):
  message = chat_input.value
  try:
    current_stage = await experiment.get_stage()
    env_text = ""
    if isinstance(current_stage, stages.EnvStage):
      timestep = current_stage.get_user_data("stage_state").timestep
      if timestep is not None:
        state = timestep.state
        if state is not None:
          # with open("env_state.txt", "w") as f:
          #    pprint.pprint(state, stream=f)
          env_text = convert_state_to_text(state)

    # Use the persisted model selection
    model = app.storage.user["selected_model"]

    if model == "gemini":
      response = await get_gemini_response(message, env_text)
    elif model == "claude":
      response = await get_claude_response(message, env_text)
    else:  # chatgpt
      response = await get_chatgpt_response(message, env_text)

    response_box.set_content(f"**Hint:** {response}")
    response_box.update()
  except Exception as e:
    response_box.set_content(f"**Error:** {str(e)}")
    response_box.update()
  chat_input.set_value("")


def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def global_handle_key_press(e, container):
  logger.info("global_handle_key_press")
  if experiment.finished():
    logger.info("Experiment finished")
    return

  stage = await experiment.get_stage()
  if stage.get_user_data("finished", False):
    return

  await stage.handle_key_press(e, container)
  local_handle_key_press = stage.get_user_data("local_handle_key_press")
  if local_handle_key_press is not None:
    await local_handle_key_press()


setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
logger = get_logger("main")

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    modules={"models": ["nicewebrl.stages"]},
  )
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
    ui.markdown("## Consent Form")
    with open("consent.md", "r") as consent_file:
      consent_text = consent_file.read()
    ui.markdown(consent_text)
    def on_change():
      print("on_change")
      consent_given.set()
    ui.checkbox("I agree to participate.", on_change=on_change)
  print("waiting for consent")
  await consent_given.wait()


async def collect_demographic_info(container):
  # Create a markdown title for the section
  nicewebrl.clear_element(container)
  collected_demographic_info_event = asyncio.Event()
  with container:
    ui.markdown("## Demographic Info")
    ui.markdown("Please fill out the following information.")

    with ui.column():
      with ui.column():
        ui.label("Biological Sex")
        sex_input = ui.radio(["Male", "Female"], value="Male").props("inline")

      # Collect age with a textbox input
      age_input = ui.input("Age")

    # Button to submit and store the data
    async def submit():
      age = age_input.value
      sex = sex_input.value

      # Validation for age input
      if not age.isdigit() or not (0 < int(age) < 100):
        ui.notify("Please enter a valid age between 1 and 99.", type="warning")
        return
      app.storage.user["age"] = int(age)
      app.storage.user["sex"] = sex
      logger.info(f"age: {int(age)}, sex: {sex}")
      collected_demographic_info_event.set()

    button = ui.button("Submit", on_click=submit)
    await button.clicked()


async def start_experiment(meta_container, stage_container, llm_container):

  # ========================================
  # Consent form and demographic info
  # ========================================
  if not (app.storage.user.get("experiment_started", False)):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

  # ========================================
  # Force fullscreen
  # ========================================
  #ui.run_javascript("window.require_fullscreen = true")

  # ========================================
  # Register global key press handler
  # ========================================
  ui.on("key_pressed", lambda e: global_handle_key_press(e, stage_container))

  # ========================================
  # LLM container
  # ========================================
  with llm_container:
    ui.markdown("## 💬 Chat with AI Assistant")
    chat_input = ui.input(placeholder="Ask for hints or clues...").style(
      "width: 100%; margin-bottom: 10px;"
    ).props('id=chat-input')
    send_button = ui.button("Send")
    response_box = ui.markdown("Waiting for your question...").style(
      "margin-top: 10px;"
    )
    send_button.on_click(lambda: send_message(chat_input, response_box))

  # ========================================
  # Start experiment
  # ========================================
  logger.info("Starting experiment")

  while not experiment.finished():
    stage = await experiment.get_stage()
    await run_stage(stage, stage_container)
    await stage.finish_saving_user_data()
    await experiment.advance_stage()

  await finish_experiment(meta_container)


async def finish_experiment(container):
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")

  #########################
  # Save data
  #########################
  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    status_container = None
    with container:
      nicewebrl.clear_element(container)
      ui.markdown(
        "## Your data is being saved. Please do not close or refresh the page."
      )
      status_container = ui.markdown("Saving local files...")

    try:
      # Create a task for the save operation with a timeout
      save_task = asyncio.create_task(save_data(feedback=feedback))
      start_time = time.time()

      # Update status every 2 seconds while waiting for save
      while not save_task.done():
        elapsed_seconds = int(time.time() - start_time)
        status_container.content = (
          f"Still saving... ({elapsed_seconds}s elapsed). This may take 5-10 minutes."
        )
        try:
          # Wait for either task completion or timeout
          await asyncio.wait_for(asyncio.shield(save_task), timeout=2.0)
        except asyncio.TimeoutError:
          # This is expected - we use timeout to update status
          continue
        except Exception as e:
          logger.error(f"Error during save: {e}")
          status_container.content = (
            "⚠️ Error saving data. Please contact the experimenter."
          )
          raise

      # If we get here, save was successful
      elapsed_seconds = int(time.time() - start_time)
      status_container.content = (
        f"✅ Save complete in {elapsed_seconds}s! Moving to next screen..."
      )
      app.storage.user["data_saved"] = True

    except Exception as e:
      logger.error(f"Save failed: {e}")
      status_container.content = "⚠️ Error saving data. Please contact the experimenter."
      raise

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with container:
      nicewebrl.clear_element(container)
      ui.markdown(
        "Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment."
      )
      text = ui.textarea().style("width: 80%;")  # Set width to 80% of the container
      button = ui.button("Submit")
      await button.clicked()
      await submit(text.value)

  #########################
  # Final screen
  #########################
  with container:
    nicewebrl.clear_element(container)
    ui.markdown("# Experiment over")
    ui.markdown("## Data saved")
    ui.markdown(
      "### Please record the following code which you will need to provide for compensation"
    )
    ui.markdown("### 'carvalho.assistants'")
    ui.markdown("#### You may close the browser")

async def save_data(feedback=None, **kwargs):
  global experiment_structure, config
  user_data_file = nicewebrl.user_data_file()
  user_metadata_file = nicewebrl.user_metadata_file()


  # --------------------------------
  # save user data to final line of file
  # --------------------------------
  user_storage = nicewebrl.make_serializable(dict(app.storage.user))
  metadata = dict(
    finished=True,
    feedback=feedback,
    user_storage=user_storage,
    **kwargs,
  )
  nicewebrl.save_metadata(metadata, user_metadata_file)

  files_to_save = [user_data_file, user_metadata_file]
  logger.info(f"Saving to bucket: {config.BUCKET_NAME}")
  await save_to_gcs_with_retries(
    files_to_save,
    max_retries=5,
    bucket_name=config.BUCKET_NAME,
  )

  # Try to delete local files after successful upload
  from nicewebrl.stages import StageStateModel
  logger.info(f"Deleting data for user {app.storage.browser['id']}")
  await StageStateModel.filter(session_id=app.storage.browser["id"]).delete()
  logger.info(
    f"Successfully deleted stage inforation for user {app.storage.browser['id']}"
  )
  for local_file in files_to_save:
    try:
      os.remove(local_file)
      logger.info(f"Successfully deleted local file: {local_file}")
    except Exception as e:
      logger.warning(f"Failed to delete local file {local_file}: {str(e)}")


async def run_stage(stage, container):
  stage_over_event = asyncio.Event()

  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via key press")
        stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  async def handle_button_press():
    if stage.get_user_data("finished", False):
      return
    await stage.handle_button_press(container)
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via button press")
        stage_over_event.set()

  with container.style("align-items: center;"):
    await stage.activate(container)

  if stage.get_user_data("finished", False):
    logger.info(f"Finished {stage.name} immediately after activation")
    stage_over_event.set()

  if stage.next_button:
    with container:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)
      await handle_button_press()

  await stage_over_event.wait()


async def check_if_over(container, episode_limit=60):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    pass


@ui.page("/")
async def index(request: Request):

  nicewebrl.initialize_user(request=request)
  await experiment.initialize()

  model_list = ["gemini", "claude", "chatgpt"]
  # Initialize random model selection if not already set
  if "selected_model" not in app.storage.user:
    rng = nicewebrl.new_rng()
    idx = jax.random.randint(rng, (), 0, len(model_list))
    app.storage.user["selected_model"] = model_list[int(idx)]

  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  card = (
    ui.card(align_items=["center"])
    .classes("fixed-center")
    .style(
      "width: 80vw;"  # Set width to 90% of viewport width
      "max-height: 90vh;"  # Keep the same max height
      "overflow: auto;"
      "display: flex;"
      "flex-direction: column;"
      "justify-content: flex-start;"
      "align-items: center;"
      "padding: 1rem;"
    )
  )
  with card:
    meta_container = ui.row()
    with meta_container.style("align-items: center;"):
      stage_container = ui.column()
      llm_container = ui.column().style("flex: 1; padding: 16px; background-color: #f5f5f5;")
      ui.timer(
        interval=10,
        callback=lambda: check_if_over(episode_limit=200, container=stage_container),
      )
      await start_experiment(meta_container, stage_container, llm_container)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload="FLY_ALLOC_ID" not in os.environ,
  title="Minigrid Web App",
  port=8080,
)
