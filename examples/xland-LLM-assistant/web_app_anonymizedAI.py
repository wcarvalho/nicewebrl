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

import experiment_structure as experiment
import config  # Import the config file

import ipdb
import pprint

DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}

# previous_obs_base64 = None
# current_obs_base64 = None


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

  model_list = ["gemini", "claude", "chatgpt"]
  # Initialize random model selection if not already set
  if "selected_model" not in app.storage.user:
    import random

    app.storage.user["selected_model"] = random.choice(model_list)

  def print_ping(e):
    logger.info(str(e.args))

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

  ui.on("ping", print_ping)
  ui.on("key_pressed", lambda e: global_handle_key_press(e, gameplay_container))

  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  with ui.row().style("width: 100vw; height: 100vh; overflow: hidden;"):
    with ui.column().style(
      "flex: 1; padding: 16px; overflow-y: auto;"
    ) as gameplay_container:
      ui.timer(
        interval=10,
        callback=lambda: check_if_over(episode_limit=200, container=gameplay_container),
      )
      asyncio.create_task(start_experiment(gameplay_container))

    with ui.column().style("flex: 1; padding: 16px; background-color: #f5f5f5;"):
      ui.markdown("## 💬 Chat with AI Assistant")
      chat_input = ui.input(placeholder="Ask for hints or clues...").style(
        "width: 100%; margin-bottom: 10px;"
      )
      send_button = ui.button("Send")
      response_box = ui.markdown("Waiting for your question...").style(
        "margin-top: 10px;"
      )

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

      async def send_message():
        message = chat_input.value
        try:
          current_stage = experiment.all_stages[1]
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

      send_button.on_click(send_message)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload=False,
  title="Minigrid Web App",
)
