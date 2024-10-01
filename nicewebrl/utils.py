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


async def wait_for_button_or_keypress(button):
    """Returns when the button is clicked or a key is pressed"""
    # Create an asyncio Future that will be set when a key is pressed
    key_pressed_future = asyncio.get_event_loop().create_future()

    # Define the keypress event handler
    def on_keypress(event):
        if not key_pressed_future.done():
            key_pressed_future.set_result(event)

    # Register the keypress event handler
    keyboard = ui.keyboard(on_key=on_keypress)

    # Await until either the button is clicked or a key is pressed
    tasks = [button.clicked(), key_pressed_future]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel any pending tasks to clean up
    for task in pending:
        if not task.done():
            task.cancel()

    # Unregister the keypress event handler to prevent memory leaks
    keyboard.delete()

    # Return the result (could be button click or keypress event)
    for task in done:
        return task.result()
