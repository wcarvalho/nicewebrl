
import functools
import random
from nicegui import ui, app
import asyncio
from nicewebrl.logging import get_logger

logger = get_logger(__name__)

async def toggle_fullscreen():
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
                logger.error(f"Error: {e}")
                continue

            finally:
                # Always try to delete the keyboard, but catch any exceptions
                try:
                    keyboard.delete()
                except Exception as e:
                    logger.error(f"{attempt}. Error deleting keyboard: {str(e)}")

        except Exception as e:
            logger.error(f"{attempt}. Waiting for button or keypress. Error occurred: {str(e)}. Retrying...")
            attempt += 1
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
