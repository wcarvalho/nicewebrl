import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import jax
import httpx

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages

import experiment_structure as experiment

DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}

previous_obs_base64 = None
current_obs_base64 = None

def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def experiment_not_finished():
  async with get_user_lock():
    not_finished = not app.storage.user.get("experiment_finished", False)
    not_finished &= app.storage.user["stage_idx"] < len(experiment.all_stages)
  return not_finished


async def global_handle_key_press(e, container):
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


async def start_experiment(container):
  if app.storage.user.get("experiment_running", False):
    return
  app.storage.user["experiment_running"] = True
  try:
    if not app.storage.user.get("experiment_started", False):
      app.storage.user["experiment_started"] = True
    logger.info("Starting experiment")
    while True and await experiment_not_finished():
      stage_idx = app.storage.user["stage_idx"]
      stage = experiment.all_stages[stage_idx]
      logger.info("=" * 30)
      logger.info(f"Began stage '{stage.name}'")
      await run_stage(stage, container)
      logger.info(f"Finished stage '{stage.name}'")
      if isinstance(stage, stages.EnvStage):
        await stage.finish_saving_user_data()
        logger.info(f"Saved data for stage '{stage.name}'")
      async with get_user_lock():
        app.storage.user["stage_idx"] = stage_idx + 1
      if app.storage.user["stage_idx"] >= len(experiment.all_stages):
        break
    await finish_experiment(container)
  finally:
    app.storage.user["experiment_running"] = False


async def finish_experiment(container):
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")


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
  user_info = dict(
    worker_id=request.query_params.get("workerId", None),
    hit_id=request.query_params.get("hitId", None),
    assignment_id=request.query_params.get("assignmentId", None),
  )

  app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
  nicewebrl.initialize_user()

  def print_ping(e):
    logger.info(str(e.args))

  ui.on("ping", print_ping)
  ui.on("key_pressed", lambda e: global_handle_key_press(e, gameplay_container))

  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  with ui.row().style("width: 100vw; height: 100vh; overflow: hidden;"):
    with ui.column().style("flex: 1; padding: 16px; overflow-y: auto;") as gameplay_container:
      ui.timer(
        interval=10,
        callback=lambda: check_if_over(
          episode_limit=200, container=gameplay_container
        ),
      )
      asyncio.create_task(start_experiment(gameplay_container))

    with ui.column().style("flex: 1; padding: 16px; background-color: #f5f5f5;"):
      ui.markdown("## 💬 Chat with Local Model")
      
      # Model selection dropdown
      model_dropdown = ui.select(
        options=[
          "gpt2",
          "facebook/opt-125m",
          "microsoft/DialoGPT-small"
        ],
        value="gpt2",
        label="Select Model"
      ).style("width: 100%; margin-bottom: 10px;")
      
      # Load model button
      load_button = ui.button("Load Model")
      model_status = ui.markdown("Model not loaded").style("margin-bottom: 10px;")
      
      chat_input = ui.input(placeholder="Ask for hints or clues...").style("width: 100%; margin-bottom: 10px;")
      send_button = ui.button("Send")
      response_box = ui.markdown("Waiting for your question...").style("margin-top: 10px;")

      # Store the loaded model
      loaded_model = None
      loaded_tokenizer = None

      async def load_model():
        nonlocal loaded_model, loaded_tokenizer
        try:
          from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
          model_name = model_dropdown.value
          model_status.set_content(f"Loading {model_name}...")
          loaded_model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
          loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
          model_status.set_content(f"Model {model_name} loaded successfully!")
        except Exception as e:
          model_status.set_content(f"Error loading model: {str(e)}")

      async def send_message():
        if not loaded_model or not loaded_tokenizer:
          response_box.set_content("Please load a model first!")
          return

        message = chat_input.value
        try:
          # Prepare the prompt with game context
          prompt = (
              "You are a helpful assistant for a Minigrid 8x8 Empty reinforcement learning game.\n"
              "The first image is the previous state of the environment.\n"
              "The second image is the current state.\n"
              "Use these to understand the user's position and goal.\n"
              "Give short, specific hints to help them progress.\n"
              "Keep responses to 1-2 lines.\n"
              f"Question: {message}\n"
              "Answer:"
          )

          if previous_obs_base64:
              prompt += f"\nPrevious state: {previous_obs_base64}\n"
          if current_obs_base64:
              prompt += f"\nCurrent state: {current_obs_base64}\n"

          
          # Tokenize and generate response using JAX
          inputs = loaded_tokenizer(prompt, return_tensors="jax")
          outputs = loaded_model.generate(
              inputs["input_ids"],
              max_length=100,
              num_return_sequences=1,
              pad_token_id=loaded_tokenizer.eos_token_id
          )
          
          response = loaded_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
          # Extract just the answer part
          answer = response.split("Answer:")[-1].strip()
          response_box.set_content(f"**Hint:** {answer}")
          response_box.update()
        except Exception as e:
          response_box.set_content(f"**Error:** {str(e)}")
          response_box.update()
        chat_input.set_value("")

      load_button.on_click(load_model)
      send_button.on_click(send_message)

ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload=False,
  title="Minigrid Web App",
)