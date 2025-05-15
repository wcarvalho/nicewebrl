import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import pdb

TILE_PIXELS = 32

# Define render function
@jax.jit
def render_fn(state):
    data = render_state(state)
    return data

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax),
            jnp.logical_and(y >= ymin, y <= ymax)
        )
    return fn

def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx)**2 + (y - cy)**2 <= r**2
    return fn

def point_in_triangle(a, b, c):
    a, b, c = jnp.array(a), jnp.array(b), jnp.array(c)
    def fn(x, y):
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
        
        result = jnp.logical_and(
            jnp.logical_and(u >= 0, v >= 0),
            (u + v) < 1
        )
        
        return result.reshape(x.shape)
    return fn

def fill_coords(img, fn, color):
    y, x = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing='ij')
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

def render_tile(agent_idx, is_goal, agent_dir_idx, tile_size=TILE_PIXELS):
    # Start with white background
    img = jnp.ones((tile_size, tile_size, 3), dtype=jnp.uint8) * 255
    
    # Add grid lines (modified to include right and bottom edges)
    fn = point_in_rect(0, 0.031, 0, 1)  # Left edge
    img = fill_coords(img, fn, jnp.array([100, 100, 100]))
    fn = point_in_rect(0.969, 1, 0, 1)  # Right edge
    img = fill_coords(img, fn, jnp.array([100, 100, 100]))
    fn = point_in_rect(0, 1, 0, 0.031)  # Top edge
    img = fill_coords(img, fn, jnp.array([100, 100, 100]))
    fn = point_in_rect(0, 1, 0.969, 1)  # Bottom edge
    img = fill_coords(img, fn, jnp.array([100, 100, 100]))
    
    # If it's a goal, fill with black
    def render_goal(img):
        fn = point_in_rect(0, 1, 0, 1)
        return fill_coords(img, fn, jnp.array([0, 0, 0]))
    
    img = jnp.where(is_goal, render_goal(img), img)
    
    # If there's an agent, render triangle
    def render_agent(img):
        # Define agent colors (red for agent 0, blue for agent 1)
        agent_colors = jnp.array([
            [255, 0, 0],    # Red for agent 0
            [0, 0, 255],    # Blue for agent 1
        ])
        
        # # Create triangle
        # tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        # # Rotate based on direction
        # tri_fn = rotate_fn(tri_fn, 0.5, 0.5, 0.5 * jnp.pi * agent_dir_idx)
        circ_fn = point_in_circle(0.5, 0.5, 0.35)
        return fill_coords(img, circ_fn, agent_colors[agent_idx])
    
    has_agent = agent_idx >= 0
    img = jnp.where(has_agent, render_agent(img), img)
    
    return img

def render_grid(agent_pos, goal_pos, agent_dir_idx, height=5, width=5, tile_size=TILE_PIXELS):
    img = jnp.zeros((height * tile_size, width * tile_size, 3), dtype=jnp.uint8)
    
    def render_tile_at(img, y, x):
        # Check if position contains an agent
        agent_at_pos = lambda agent_p: jnp.all(jnp.array([x, y]) == agent_p)
        is_agent_0 = agent_at_pos(agent_pos[0])
        is_agent_1 = agent_at_pos(agent_pos[1])
        agent_idx = jnp.where(is_agent_0, 0, jnp.where(is_agent_1, 1, -1))  # -1 if no agent
        
        # Check if position is a goal
        on_goal = lambda pos1, pos2: jnp.all(pos1 == pos2)
        on_any_goal = lambda x: jnp.any(jax.vmap(on_goal, in_axes=(None, 0))(x, goal_pos))
        is_goal = on_any_goal(jnp.array([x, y]))
        
        # Get direction if there's an agent
        dir_idx = jnp.where(is_agent_0, agent_dir_idx[0], 
                           jnp.where(is_agent_1, agent_dir_idx[1], 0))
        
        tile_img = render_tile(agent_idx, is_goal, dir_idx).astype(jnp.uint8)
        return lax.dynamic_update_slice(img, tile_img, (y*tile_size, x*tile_size, jnp.array(0, dtype=jnp.int32)))
    
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
def render_state(state, tile_size=TILE_PIXELS):
    img = render_grid(
        state.agent_pos,
        state.goal_pos,
        jnp.zeros_like(state.agent_pos[:, 0], dtype=jnp.int32),  # No direction in toy_coop
        height=5,
        width=5,
    )
    return img
