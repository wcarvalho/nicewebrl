import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

from jaxmarl.environments.overcooked.common import (
  OBJECT_TO_INDEX,
  COLOR_TO_INDEX,
  COLORS,
)
import pdb

TILE_PIXELS = 32
INDEX_TO_COLOR = [k for k, v in COLOR_TO_INDEX.items()]

JAX_COLORS = np.stack([COLORS[k] for k in INDEX_TO_COLOR])
JAX_COLORS = jnp.array(JAX_COLORS)

COLOR_TO_AGENT_INDEX = {0: 0, 2: 1}  # Hardcoded. Red is first, blue is second

# JAX-compatible utility functions (originally from grid_rendering.py)

# Define render function


def render_fn(state):
  data = render_state(state, highlight=False)
  return data


def point_in_rect(xmin, xmax, ymin, ymax):
  def fn(x, y):
    return jnp.logical_and(
      jnp.logical_and(x >= xmin, x <= xmax), jnp.logical_and(y >= ymin, y <= ymax)
    )

  return fn


def point_in_circle(cx, cy, r):
  def fn(x, y):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2

  return fn


def point_in_triangle(a, b, c):
  a, b, c = jnp.array(a), jnp.array(b), jnp.array(c)

  def fn(x, y):
    # Reshape x and y into 2D arrays of shape (2, 32*32)
    points = jnp.stack([x.ravel(), y.ravel()], axis=0)

    v0 = c - a
    v1 = b - a
    v2 = points - a[:, None]

    dot00 = jnp.dot(v0, v0)
    dot01 = jnp.dot(v0, v1)
    dot02 = jnp.sum(v0[:, None] * v2, axis=0)
    dot11 = jnp.dot(v1, v1)
    dot12 = jnp.sum(v1[:, None] * v2, axis=0)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    result = jnp.logical_and(jnp.logical_and(u >= 0, v >= 0), (u + v) < 1)

    # Reshape the result back to 32x32
    return result.reshape(x.shape)

  return fn


def fill_coords(img, fn, color):
  y, x = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing="ij")
  yf = (y + 0.5) / img.shape[0]
  xf = (x + 0.5) / img.shape[1]
  mask = fn(xf, yf)
  return jnp.where(mask[:, :, None], color, img)


def rotate_fn(fin, cx, cy, theta):
  def fout(x, y):
    x = x - cx
    y = y - cy
    x2 = cx + x * jnp.cos(-theta) - y * jnp.sin(-theta)
    y2 = cy + y * jnp.cos(-theta) + x * jnp.sin(-theta)
    return fin(x2, y2)

  return fout


def render_tile(obj, highlight, agent_dir_idx, agent_inv, tile_size=TILE_PIXELS):
  # img = jnp.ones((tile_size, tile_size, 3), dtype=jnp.uint8) * 255
  img = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.uint8)

  # Render grid lines
  fn = point_in_rect(0, 0.031, 0, 1)
  # img = fill_coords(img, fn, jnp.array([255, 255, 255]))
  img = fill_coords(img, fn, jnp.array([100, 100, 100]))
  fn = point_in_rect(0, 1, 0, 0.031)
  # img = fill_coords(img, fn, jnp.array([255, 255, 255]))
  img = fill_coords(img, fn, jnp.array([100, 100, 100]))

  def render_obj(img, obj_type, color):
    def do_nothing(x):
      return x

    def render_wall(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, color)
      return img

    def render_goal(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])
      fn = point_in_rect(0.1, 0.9, 0.1, 0.9)
      img = fill_coords(img, fn, color)
      return img

    def render_agent(img):
      tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
      agent_idx_dir_map = jnp.array([3, 1, 0, 2], dtype=jnp.int32)
      tri_fn = rotate_fn(
        tri_fn, 0.5, 0.5, 0.5 * jnp.pi * agent_idx_dir_map[agent_dir_idx]
      )
      img = fill_coords(img, tri_fn, color)

      # Render inventory item
      def render_inventory(img):
        def _render_nothing(x):
          return x

        def _render_mini_onion(img):
          # if inv_type == OBJECT_TO_INDEX['onion']:
          onion_fn = point_in_circle(0.75, 0.75, 0.15)
          img = fill_coords(img, onion_fn, COLORS["yellow"])
          return img

        def _render_mini_plate(img):
          # elif inv_type == OBJECT_TO_INDEX['plate']:
          plate_fn = point_in_circle(0.75, 0.75, 0.2)
          img = fill_coords(img, plate_fn, COLORS["white"])
          return img

        def _render_mini_dish(img):
          # elif inv_type == OBJECT_TO_INDEX['dish']:
          plate_fn = point_in_circle(0.75, 0.75, 0.2)
          img = fill_coords(img, plate_fn, COLORS["white"])
          onion_fn = point_in_circle(0.75, 0.75, 0.13)
          img = fill_coords(img, onion_fn, COLORS["orange"])
          return img

        img = lax.cond(
          agent_inv == OBJECT_TO_INDEX["onion"],
          _render_mini_onion,
          _render_nothing,
          img,
        )
        img = lax.cond(
          agent_inv == OBJECT_TO_INDEX["plate"],
          _render_mini_plate,
          _render_nothing,
          img,
        )
        img = lax.cond(
          agent_inv == OBJECT_TO_INDEX["dish"],
          _render_mini_dish,
          _render_nothing,
          img,
        )

        return img

      img = lax.cond(agent_inv != 1, render_inventory, lambda x: x, img)
      return img

    def render_empty(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["black"])
      return img

    def render_onion_pile(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])
      onion_coords = jnp.array(
        [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35), (0.4, 0.8), (0.75, 0.75)]
      )

      def apply_onion(carry, coord_id):
        img, onion_coords = carry
        coord = onion_coords[coord_id]
        onion_fn = point_in_circle(*coord, 0.15)
        return (fill_coords(img, onion_fn, color), onion_coords), None

      carry, _ = jax.lax.scan(
        apply_onion, (img, onion_coords), jnp.arange(len(onion_coords))
      )
      img = carry[0]
      return img

    def render_onion(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])
      onion_fn = point_in_circle(0.5, 0.5, 0.15)
      img = fill_coords(img, onion_fn, color)
      return img

    def render_plate_pile(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])

      def apply_plate(carry, coord_id):
        img, plate_coords = carry
        coord = plate_coords[coord_id]
        plate_fn = point_in_circle(*coord, 0.2)
        return (fill_coords(img, plate_fn, color), plate_coords), None

      plate_coords = jnp.array([(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)])
      carry, _ = jax.lax.scan(
        apply_plate, (img, plate_coords), jnp.arange(len(plate_coords))
      )
      img = carry[0]
      return img

    def render_plate(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])
      plate_fn = point_in_circle(0.5, 0.5, 0.2)
      img = fill_coords(img, plate_fn, color)
      return img

    def render_dish(img):
      fn = point_in_rect(0, 1, 0, 1)
      img = fill_coords(img, fn, COLORS["grey"])
      plate_fn = point_in_circle(0.5, 0.5, 0.2)
      img = fill_coords(img, plate_fn, color)
      onion_fn = point_in_circle(0.5, 0.5, 0.13)
      img = fill_coords(img, onion_fn, color)
      return img

    def render_pot(img):
      img = rendering_pot(obj, img)
      return img

    img = lax.cond(obj_type == OBJECT_TO_INDEX["wall"], render_wall, do_nothing, img)

    img = lax.cond(obj_type == OBJECT_TO_INDEX["goal"], render_goal, do_nothing, img)
    img = lax.cond(obj_type == OBJECT_TO_INDEX["agent"], render_agent, do_nothing, img)
    img = lax.cond(obj_type == OBJECT_TO_INDEX["empty"], render_empty, do_nothing, img)
    img = lax.cond(
      obj_type == OBJECT_TO_INDEX["onion_pile"],
      render_onion_pile,
      do_nothing,
      img,
    )
    img = lax.cond(obj_type == OBJECT_TO_INDEX["onion"], render_onion, do_nothing, img)
    img = lax.cond(
      obj_type == OBJECT_TO_INDEX["plate_pile"],
      render_plate_pile,
      do_nothing,
      img,
    )
    img = lax.cond(obj_type == OBJECT_TO_INDEX["plate"], render_plate, do_nothing, img)
    img = lax.cond(obj_type == OBJECT_TO_INDEX["dish"], render_dish, do_nothing, img)
    img = lax.cond(obj_type == OBJECT_TO_INDEX["pot"], render_pot, do_nothing, img)

    return img

  img = lax.cond(
    obj is not None,
    lambda img: render_obj(img, obj[0], JAX_COLORS[obj[1]]),
    lambda img: img,
    img,
  )

  return img


def rendering_pot(obj, img):
  pot_status = obj[-1]
  num_onions = jnp.maximum(23 - pot_status, 0)
  is_cooking = (pot_status < 20) & (pot_status > 0)
  is_done = pot_status == 0

  pot_fn = point_in_rect(0.1, 0.9, 0.33, 0.9)
  lid_fn = point_in_rect(0.1, 0.9, 0.21, 0.25)
  handle_fn = point_in_rect(0.4, 0.6, 0.16, 0.21)

  img = fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS["grey"])

  def render_onions(img, num_onions):
    onion_fns = jnp.array([(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)])

    def render_onion(carry, i):
      img, coords, num_onions = carry
      coord = coords[i]
      onion_fn = point_in_circle(*coord, 0.13)
      img = jnp.where(
        i < num_onions, fill_coords(img, onion_fn, COLORS["yellow"]), img
      )  # show progress of onions filled
      return (img, coords, num_onions), None

    carry, _ = lax.scan(
      render_onion, (img, onion_fns, num_onions), jnp.arange(len(onion_fns))
    )
    img = carry[0]
    return img

  img = lax.cond(
    (num_onions > 0) & ~is_done, render_onions, lambda x, y: x, img, num_onions
  )

  img = lax.cond(
    is_done,
    lambda x: fill_coords(x, point_in_rect(0.12, 0.88, 0.23, 0.35), COLORS["orange"]),
    lambda x: x,
    img,
  )
  img = fill_coords(img, pot_fn, COLORS["black"])
  img = fill_coords(img, lid_fn, COLORS["black"])
  img = fill_coords(img, handle_fn, COLORS["black"])

  img = lax.cond(
    is_cooking,
    lambda x: fill_coords(
      x,
      point_in_rect(0.1, 0.9 - (0.9 - 0.1) / 20 * pot_status, 0.83, 0.88),
      COLORS["green"],
    ),
    lambda x: x,
    img,
  )

  return img


def render_grid(grid, highlight_mask, agent_dir_idx, agent_inv, tile_size=TILE_PIXELS):
  height, width = grid.shape[:2]
  img = jnp.zeros((height * tile_size, width * tile_size, 3), dtype=jnp.uint8)

  def render_tile_at(img, y, x):
    obj = grid[y, x]
    agent_id = jnp.where(obj[1] == 2, 1, 0)  # 2 is blue, 0 is red
    tile_img = render_tile(
      obj, highlight_mask[y, x], agent_dir_idx[agent_id], agent_inv[agent_id]
    ).astype(jnp.uint8)
    return lax.dynamic_update_slice(
      img, tile_img, (y * tile_size, x * tile_size, jnp.array(0, dtype=jnp.int32))
    )

  def render_row(carry, x):
    img, y = carry
    img = render_tile_at(img, y, x)
    return (img, y), None

  def render_grid_rows(img, y):
    (img, y), _ = lax.scan(render_row, (img, y), jnp.arange(width))
    return img, None

  img, _ = lax.scan(render_grid_rows, img, jnp.arange(height))

  return img


@jax.jit
def render_state(state, highlight=False, tile_size=TILE_PIXELS, agent_view_size=5):
  padding = agent_view_size - 2
  grid = state.maze_map[padding:-padding, padding:-padding, :]

  highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)

  img = render_grid(
    grid,
    highlight_mask,
    state.agent_dir_idx,
    state.agent_inv,
  )

  return img
