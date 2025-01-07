
import functools
import random
from nicegui import ui, app
import asyncio
from nicewebrl.logging import get_logger
import jax
import jax.numpy as jnp
from nicegui import app
import os.path
import random
from datetime import datetime

logger = get_logger(__name__)

async def toggle_fullscreen():
  logger.info("Toggling fullscreen")
  await ui.run_javascript('''
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
    ''', timeout=10)


async def check_fullscreen():
    result = None
    try:
        result = await ui.run_javascript(
            'return document.fullscreenElement !== null;',
            timeout=10)
    except TimeoutError as e:
        # Handle the timeout, maybe retry or log the error
        logger.error(f'JavaScript execution timed out: {e}')
    return result

def clear_element(element):
    try:
        element.clear()
    except Exception as e:
        logger.error(f'Error clearing element: {e}')

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
                    if (current_time - last_key_press_time) > .5 and not key_pressed_future.done():
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
            logger.error(f"{attempt}. Waiting for button or keypress. Error occurred: '{str(e)}'. Retrying...")
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
                    
                    delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1), max_delay)
                    logger.error(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
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
        app.storage.user['seed'] = seed
    else:
        app.storage.user['seed'] = app.storage.user.get(
            'seed', random.getrandbits(32))

    app.storage.user['rng_splits'] = app.storage.user.get('rng_splits', 0)
    if 'rng_key' not in app.storage.user:
        rng_key = jax.random.PRNGKey(app.storage.user['seed'])
        app.storage.user['rng_key'] = rng_key.tolist()
        app.storage.user['init_rng_key'] = app.storage.user['rng_key']
    app.storage.user['session_start'] = app.storage.user.get(
        'session_start',
        datetime.now().isoformat())
    app.storage.user['session_duration'] = 0


def get_user_session_minutes():
    start_time = datetime.fromisoformat(app.storage.user['session_start'])
    current_time = datetime.now()
    duration = current_time - start_time
    minutes_passed = duration.total_seconds() / 60
    app.storage.user['session_duration'] = minutes_passed
    return minutes_passed

@ui.refreshable
def broadcast_message(event: str, message: str):
  called_by_user_id = str(app.storage.user['seed'])
  called_by_room_id = str(app.storage.user['room_id'])
  stage = app.storage.user['stage_idx']
  fn = f"userMessage('{called_by_room_id}', '{called_by_user_id}', '{event}', '{stage}', '{message}')"
  logger.info(fn)
  ui.run_javascript(fn)
