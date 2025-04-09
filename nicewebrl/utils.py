from typing import List, Dict
import aiofiles
import functools
import random
from nicegui import ui, app, Client
import asyncio
from nicewebrl.logging import get_logger
import jax
import jax.numpy as jnp
import msgpack
import os.path
import random
from datetime import datetime
import struct

logger = get_logger(__name__)

_user_locks = {}

def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = asyncio.Lock()
  return _user_locks[user_seed]

async def toggle_fullscreen():
  logger.info("Toggling fullscreen")
  await ui.run_javascript(
    """
    return (async () => {
        if (!document.fullscreenElement) {
            await document.documentElement.requestFullscreen();
        } else {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            }
        }
        return true;
    })();
    """,
    timeout=10,
  )


async def check_fullscreen():
  result = None
  try:
    result = await ui.run_javascript(
      "return document.fullscreenElement !== null;", timeout=10
    )
  except TimeoutError as e:
    # Handle the timeout, maybe retry or log the error
    logger.error(f"JavaScript execution timed out: {e}")
  return result


def clear_element(element):
  try:
    element.clear()
  except Exception as e:
    logger.error(f"Error clearing element: {e}")


async def wait_for_button_or_keypress(button, ignore_recent_press=False):
  attempt = 0
  while True:  # This will keep the function running indefinitely
    try:
      key_pressed_future = asyncio.get_event_loop().create_future()
      last_key_press_time = asyncio.get_event_loop().time()

      def on_keypress(event):
        nonlocal last_key_press_time
        current_time = asyncio.get_event_loop().time()

        if ignore_recent_press:
          if (
            current_time - last_key_press_time
          ) > 0.5 and not key_pressed_future.done():
            key_pressed_future.set_result(event)
        else:
          if not key_pressed_future.done():
            key_pressed_future.set_result(event)

        last_key_press_time = current_time

      keyboard = ui.keyboard(on_key=on_keypress)

      try:
        tasks = [button.clicked(), key_pressed_future]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
          if not task.done():
            task.cancel()

        for task in done:
          return task.result()

      except asyncio.CancelledError as e:
        logger.error(f"{attempt}. Task was cancelled. Cleaning up and retrying...")
        logger.error(f"Error: '{e}'")
        continue

      finally:
        # Always try to delete the keyboard, but catch any exceptions
        try:
          keyboard.delete()
        except Exception as e:
          logger.error(f"{attempt}. Error deleting keyboard: '{str(e)}'")

    except Exception as e:
      logger.error(
        f"{attempt}. Waiting for button or keypress. Error occurred: '{str(e)}'. Retrying..."
      )
      attempt += 1
      if attempt > 10:
        return
      await asyncio.sleep(1)  # Add a small delay before retrying


def retry_with_exponential_backoff(max_retries=3, base_delay=1, max_delay=10):
  def decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
      for attempt in range(1, max_retries + 1):
        try:
          return await func(*args, **kwargs)
        except Exception as e:
          if attempt == max_retries:
            logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
            raise

          delay = min(
            base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1),
            max_delay,
          )
          logger.error(
            f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f} seconds..."
          )
          await asyncio.sleep(delay)

      # This line should never be reached, but it's here for completeness
      raise Exception("Unexpected error in retry logic")

    return wrapper

  return decorator


def basic_javascript_file():
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  file = f"{current_directory}/basics.js"
  return file


def multihuman_javascript_file():
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  file = f"{current_directory}/multihuman_basics.js"
  return file


def initialize_user(seed: int = 0, *kwargs):
  """
  Initialize user-specific data and settings.

  Args:
      seed (int, optional): The seed for random number generation.
          Defaults to 0.
  """
  if seed:
    app.storage.user["seed"] = seed
  else:
    app.storage.user["seed"] = app.storage.user.get("seed", random.getrandbits(32))

  app.storage.user["rng_splits"] = app.storage.user.get("rng_splits", 0)
  if "rng_key" not in app.storage.user:
    rng_key = jax.random.PRNGKey(app.storage.user["seed"])
    app.storage.user["rng_key"] = rng_key.tolist()
    app.storage.user["init_rng_key"] = app.storage.user["rng_key"]
  app.storage.user["session_start"] = app.storage.user.get(
    "session_start", datetime.now().isoformat()
  )
  app.storage.user["session_duration"] = 0


def get_user_session_minutes():
  start_time = datetime.fromisoformat(app.storage.user["session_start"])
  current_time = datetime.now()
  duration = current_time - start_time
  minutes_passed = duration.total_seconds() / 60
  app.storage.user["session_duration"] = minutes_passed
  return minutes_passed


def broadcast_message(event: str, message: str):
  called_by_user_id = str(app.storage.user["seed"])
  called_by_room_id = str(app.storage.user["room_id"])
  stage = app.storage.user["stage_idx"]
  fn = f"userMessage('{called_by_room_id}', '{called_by_user_id}', '{event}', '{stage}', '{message}')"
  logger.info(fn)
  for client in Client.instances.values():
    with client:
      ui.run_javascript(fn)


async def write_msgpack_record(f, data):
  """Write a length-prefixed msgpack record to a file.

  Args:
      f: An aiofiles file object opened in binary mode
      data: The data to write
  """
  packed_data = msgpack.packb(data)
  length = len(packed_data)
  await f.write(length.to_bytes(4, byteorder="big"))
  await f.write(packed_data)


async def read_msgpack_records(filepath: str):
  """Read length-prefixed msgpack records from a file.

  Args:
      filepath: Path to the file containing the records

  Yields:
      Decoded msgpack records one at a time
  """
  async with aiofiles.open(filepath, "rb") as f:
    while True:
      # Read length prefix (4 bytes)
      length_bytes = await f.read(4)
      if not length_bytes:  # End of file
        break

      # Convert bytes to integer
      length = int.from_bytes(length_bytes, byteorder="big")

      # Read the record data
      data = await f.read(length)
      if len(data) < length:  # Incomplete record
        logger.error(
          f"Corrupt data in {filepath}: Expected {length} bytes but got {len(data)}"
        )
        # break

      # Unpack and yield the record
      try:
        record = msgpack.unpackb(data, strict_map_key=False)
        yield record
      except Exception as e:
        logger.error(f"Failed to unpack record in {filepath}: {e}")
        break


def read_msgpack_records_sync(filepath: str):
  """Synchronous version of read_msgpack_records that reads msgpack records from a file."""
  try:
    with open(filepath, "rb") as f:
      # Read the file content
      content = f.read()

      # Initialize position
      pos = 0

      # Read records until we reach the end of the file
      while pos < len(content):
        # Read the size of the next record
        size_bytes = content[pos : pos + 4]
        if len(size_bytes) < 4:
          break

        size = struct.unpack(">I", size_bytes)[0]
        pos += 4

        # Read the record data
        if pos + size > len(content):
          logger.error(f"Incomplete record in {filepath}")
          break

        data = content[pos : pos + size]
        pos += size

        # Unpack and yield the record
        try:
          record = msgpack.unpackb(data, strict_map_key=False)
          yield record
        except Exception as e:
          logger.error(f"Failed to unpack record in {filepath}: {e}")
          break
  except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
  except Exception as e:
    logger.error(f"Error reading {filepath}: {e}")


async def read_all_records(filepath: str) -> List[Dict]:
  """Helper function to read all records into a list."""
  return [record async for record in read_msgpack_records(filepath)]


def read_all_records_sync(filepath: str) -> List[Dict]:
  """Synchronous version that reads all msgpack records from a file and returns them as a list."""
  return list(read_msgpack_records_sync(filepath))
