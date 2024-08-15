from nicegui import ui

async def toggle_fullscreen():
  await ui.run_javascript('''
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
    ''')


async def check_fullscreen():
    result = await ui.run_javascript('isFullscreen()')
    return result
