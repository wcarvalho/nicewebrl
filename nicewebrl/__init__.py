import jax
import jax.numpy as jnp
from nicegui import app
import os.path
import random

def basic_javascript_file():
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  file = f"{current_directory}/basics.js"
  return file


def initialize_user():
    app.storage.user['seed'] = app.storage.user.get(
        'seed', random.getrandbits(32))
    app.storage.user['rng_splits'] = app.storage.user.get('rng_splits', 0)
    app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)
    if 'rng_key' not in app.storage.user:
        rng_key = jax.random.PRNGKey(app.storage.user['seed'])
        app.storage.user['rng_key'] = rng_key.tolist()
