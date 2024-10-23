import io
from nicegui import ui
import asyncio

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
        print(f'JavaScript execution timed out: {e}')
    return result


async def wait_for_button_or_keypress(button, ignore_recent_press=False):
    """Returns when the button is clicked or a new key is pressed"""
    key_pressed_future = asyncio.get_event_loop().create_future()
    last_key_press_time = asyncio.get_event_loop().time()  # Initialize with current time

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

    tasks = [button.clicked(), key_pressed_future]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        if not task.done():
            task.cancel()

    keyboard.delete()

    for task in done:
        return task.result()


class TeeOutput(io.TextIOBase):
    def __init__(self, file_stream, console_stream):
        self.file_stream = file_stream
        self.console_stream = console_stream

    def write(self, s):
        self.file_stream.write(s)
        self.console_stream.write(s)
        return len(s)

    def flush(self):
        self.file_stream.flush()
        self.console_stream.flush()
