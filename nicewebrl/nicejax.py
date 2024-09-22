import time
import typing
from typing import Union, Any, Callable
from typing import get_type_hints
from base64 import b64encode, b64decode
from flax import struct
from flax import serialization
import io
import inspect
import jax.numpy as jnp
import jax.random
import numpy as np
import random
from nicegui import app, ui
from PIL import Image

Timestep = Any
RenderFn = Callable[[Timestep], jax.Array]

def get_rng():
    """Initializes a jax.random number generator or gets the latest if already initialized."""
    app.storage.user['seed'] = app.storage.user.get(
        'seed', random.getrandbits(32))
    app.storage.user['rng_splits'] = app.storage.user.get('rng_splits', 0)
    if 'rng_key' in app.storage.user:
        rng_key = jnp.array(
            app.storage.user['rng_key'], dtype=jax.numpy.uint32)
        return rng_key
    else:
        rng_key = jax.random.PRNGKey(app.storage.user['seed'])
        app.storage.user['rng_key'] = rng_key.tolist()
        return rng_key

def new_rng():
    """Return a new jax.random number generator or make a new one if not initialized."""
    app.storage.user['seed'] = app.storage.user.get('seed', random.getrandbits(32))
    if 'rng_key' in app.storage.user:
        rng_key = jnp.array(
            app.storage.user['rng_key'], dtype=jax.numpy.uint32)
        rng_key, rng = jax.random.split(rng_key)
        app.storage.user['rng_key'] = rng_key.tolist()
        app.storage.user['rng_splits'] = app.storage.user.get('rng_splits', 0) + 1
        return rng
    else:
        rng_key = jax.random.PRNGKey(app.storage.user['seed'])
        app.storage.user['rng_key'] = rng_key.tolist()
        return rng_key


def match_types(example: struct.PyTreeNode, data: struct.PyTreeNode):
    def match_types_(ex, d):
        if d is None:
            return None
        else:
            if hasattr(ex, 'dtype'):
                return jax.numpy.array(d, dtype=ex.dtype)
            else:
                return d
    return jax.tree_map(match_types_, example, data)

def make_serializable(obj: Any):
    """Convert nested jax objects to serializable python objects"""
    if isinstance(obj, np.ndarray):
      return obj.tolist()  # Convert JAX array to list
    elif isinstance(obj, jnp.ndarray):
      obj = jax.tree_map(np.array, obj)
      return obj.tolist()  # Convert JAX array to list
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    else:
        return obj

def deserialize(cls: struct.PyTreeNode, data: Any):
    """
    Automatically deserialize data into the given class.
    
    Args:
    cls: The class to deserialize into
    data: The data to deserialize
    
    Returns:
    An instance of cls with deserialized data
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, int):
        try:
            return jnp.array(data, dtype=jnp.int32)
        except OverflowError:
            return jnp.array(data, dtype=jnp.uint32)
    elif isinstance(data, float):
        return jnp.array(data, dtype=jnp.float32)
    elif isinstance(data, bool):
        return jnp.array(data, dtype=jnp.bool_)

    if isinstance(data, jnp.ndarray):
        return data

    if isinstance(data, list):
        return [deserialize(cls, item) for item in data]

    if cls == jnp.ndarray:
        is_dict = isinstance(data, dict)
        integer_keys = all(k.isdigit() for k in data.keys())
        #number_values = all(
            #isinstance(v, (int, float)) for v in data.values())
        if is_dict and integer_keys:
          array = [deserialize(cls, data[str(i)]) for i in range(len(data))]
          try:
            return jnp.array(array)
          except OverflowError:
                if isinstance(data['0'], int):
                    return jnp.array(array, dtype=jnp.uint32)
                else:
                    raise RuntimeError("how to handle?")
        else:
            raise RuntimeError("malformed numpy array?")

    if isinstance(data, dict):
        hints = get_type_hints(cls)
        kwargs = {}
        for key, value in data.items():
            if key in hints:
                field_type = hints[key]
                # Check if the field_type is Optional
                if typing.get_origin(field_type) is typing.Union and type(None) in typing.get_args(field_type):
                    # It's an Optional type
                    if value is None:
                        kwargs[key] = None
                    else:
                        # Get the type inside Optional
                        inner_type = next(t for t in typing.get_args(
                            field_type) if t is not type(None))
                        kwargs[key] = deserialize(inner_type, value)
                elif inspect.isclass(field_type) and (
                        issubclass(field_type, struct.PyTreeNode) 
                        or hasattr(field_type, '__annotations__')):
                    kwargs[key] = deserialize(field_type, value)

                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value
        return cls(**kwargs)

    raise ValueError(f"Unable to deserialize {data} into {cls}")

def deserialize_bytes(
        cls: struct.PyTreeNode,
        encoded_data: bytes,
):
    data_bytes = b64decode(encoded_data)
    deserialized_tree = serialization.from_bytes(
        None, data_bytes)
    deserialized = deserialize(cls, deserialized_tree)
    return deserialized

def base64_npimage(image: np.ndarray):
    image = np.asarray(image)
    buffer = io.BytesIO()
    Image.fromarray(image.astype('uint8')).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded_image

class JaxWebEnv:
    def __init__(self, env):
        self.env = env
        assert hasattr(env, 'reset'), 'env needs reset function'
        assert hasattr(env, 'step'), 'env needs step function'

        num_actions = env.num_actions()
        def next_steps(rng, timestep, env_params):
            actions = jnp.arange(num_actions)
            rngs = jax.random.split(rng, num_actions)

            # vmap over rngs and actions. re-use timestep
            timesteps = jax.vmap(
                env.step, in_axes=(0, None, 0, None), out_axes=0
            )(rngs, timestep, actions, env_params)
            return timesteps

        self.reset = jax.jit(self.env.reset)
        self.next_steps = jax.jit(next_steps)

    def precompile(self, dummy_env_params: struct.PyTreeNode) -> None:
        """Call this function to pre-compile jax functions before experiment starts."""
        print("Compiling jax environment functions.")
        start = time.time()
        dummy_rng = jax.random.PRNGKey(0)
        self.reset = self.reset.lower(dummy_rng, dummy_env_params).compile()
        timestep = self.reset(dummy_rng, dummy_env_params)
        self.next_steps = self.next_steps.lower(dummy_rng, timestep, dummy_env_params).compile()
        print("\ttime:", time.time()-start)

    def precompile_vmap_render_fn(
            self,
            render_fn: RenderFn,
            dummy_env_params: struct.PyTreeNode) -> RenderFn:
        """Call this function to pre-compile a multi-render function before experiment starts."""
        print("Compiling multi-render function.")
        start = time.time()
        vmap_render_fn = jax.jit(jax.vmap(render_fn))
        dummy_rng = jax.random.PRNGKey(0)
        timestep = self.reset(dummy_rng, dummy_env_params)
        next_timesteps = self.next_steps(
            dummy_rng, timestep, dummy_env_params)
        vmap_render_fn = vmap_render_fn.lower(
            next_timesteps).compile()
        print("\ttime:", time.time()-start)
        return vmap_render_fn
