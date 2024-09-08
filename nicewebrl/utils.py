from nicegui import ui

async def toggle_fullscreen():
  await ui.run_javascript('''
    if (!document.fullscreenElement) {
        return document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) {
            return document.exitFullscreen();
        }
    }
    ''',
    timeout=10)


async def check_fullscreen():
    result = await ui.run_javascript('isFullscreen()')
    return result
