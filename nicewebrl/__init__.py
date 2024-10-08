import jax
import jax.numpy as jnp
from nicegui import app
import os.path
import random
from datetime import datetime

def basic_javascript_file():
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  file = f"{current_directory}/basics.js"
  return file


def initialize_user(debug: bool = False, debug_seed: int = 42):
    if debug:
        app.storage.user['seed'] = debug_seed
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
