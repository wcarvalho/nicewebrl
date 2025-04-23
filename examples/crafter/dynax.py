import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import flax.struct as struct
from flax.training.train_state import TrainState
import optax
import flashbax as fbx
import functools
import numpy as np
from typing import Optional, Any
from craftax.craftax_env import make_craftax_env_from_name
from nicewebrl.nicejax import TimestepWrapper
from jaxneurorl import losses
import rlax

# Assume gymnax environment imports

# --- Dataclasses ---

# for single env, single timestep
class StepType(jnp.uint8):
  FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
  MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
  LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


class TimeStep(struct.PyTreeNode):
  state: struct.PyTreeNode

  step_type: StepType
  reward: jax.Array
  discount: jax.Array
  observation: jax.Array

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):
    return self.step_type == StepType.LAST

@struct.dataclass
class Transition:
    # Structure stored in the replay buffer
    timestep: TimeStep      # Information at time t (s_t, r_t, done_t)
    action: jax.Array       # Action taken at time t (a_t)
    agent_state: jax.Array # Agent's RNN state *before* processing timestep t (h_{t-1})

@struct.dataclass
class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

@struct.dataclass
class RunnerState():
    train_state: CustomTrainState  # Contains network params, optimizer state, etc.
    env_timestep: TimeStep         # Current environment state
    agent_state: jax.Array        # Current RNN state of the agent
    buffer_state: Any              # State of the replay buffer
    rng: jax.Array                # Random number generator key

@struct.dataclass
class Predictions:
    q_vals: jax.Array
    state: struct.PyTreeNode
    # Optionally store other network outputs if needed

@struct.dataclass
class SimulationOutput:
  actions: jax.Array
  predictions: Optional[Predictions] = None

# --- Network Definition ---
# Based on ScannedRNN from PureJaxRL
class DynaAgent(nn.Module):
    config: dict
    env: TimestepWrapper
    env_params: Any

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        hidden_size = self.config["RNN_HIDDEN_DIM"]
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], hidden_size),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        preds = Predictions(q_vals=y, state=new_rnn_state)
        return new_rnn_state, preds

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )
    
    # NOTE: changed signature to be more general -> state is now a struct.PyTreeNode
    def apply_world_model(self, state: struct.PyTreeNode, action: jax.Array, rng: jax.Array) -> struct.PyTreeNode:
        """
        Simulates one step using the 'world model' (ground truth env).
        """
        # vmap the step function
        vmap_step = lambda n_envs: lambda rng, state, action: jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng, n_envs), state, action, self.env_params)
        
        # Implement ground truth env.step call
        # TODO: fix outputs to be TimeStep object
        next_state = vmap_step(self.config["N_ENVS"])(rng, state, action)
        return next_state

# --- Loss Function ---

def make_float(x):
  return x.astype(jnp.float32)

def add_time(v):
  return jax.tree.map(lambda x: x[None], v)

def is_truncated(timestep):
  non_terminal = timestep.discount

  # either termination or truncation
  is_last = make_float(timestep.last())

  # non_terminal AND is_last confirms truncation
  truncated = (non_terminal + is_last) > 1
  return make_float(1 - truncated)

def concat_pytrees(tree1, tree2, **kwargs):
  return jax.tree.map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)

def concat_first_rest(first, rest):
  first = add_time(first)  # [N, ...] --> [1, N, ...]
  # rest: [T, N, ...]
  # output: [T+1, N, ...]
  return jax.vmap(concat_pytrees, 1, 1)(first, rest)

def concat_start_sims(start, simulations):
  # concat where vmap over simulation dimension
  # need this since have 1 start path, but multiple simulations
  concat_ = lambda a, b: jnp.concatenate((a, b))
  concat_ = jax.vmap(concat_, (None, 1), 1)
  return jax.tree.map(concat_, start, simulations)

@struct.dataclass
class DynaLossFn:
    agent: DynaAgent
    config: dict # Containing GAMMA, TD_LAMBDA, ONLINE_COEFF, DYNA_COEFF, SIM_LENGTH, etc.
    simulation_policy_fn: callable # Function: (q_vals, rng) -> actionp

    def batch_loss(
        self,
        timestep,       # Includes non_terminal etc. [T+1, B, ...]
        online_preds,   # Includes q_vals [T+1, B, A]
        target_preds,   # Includes q_vals [T+1, B, A]
        actions,        # [T+1, B]
        rewards,        # [T+1, B]
        non_terminal,   # [T+1, B]
        is_last,        # [T+1, B]
        loss_mask,      # [T+1, B]
    ):
        
        rewards = make_float(rewards)
        rewards = rewards - self.step_cost
        is_last = make_float(is_last)
        discounts = make_float(non_terminal) * self.discount
        lambda_ = jnp.ones_like(non_terminal) * self.lambda_

        # Get N-step transformed TD error and loss.
        batch_td_error_fn = jax.vmap(losses.q_learning_lambda_td, in_axes=1, out_axes=1)

        # [T, B]
        selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)  # [T+1, B]
        q_t, target_q_t = batch_td_error_fn(
        online_preds.q_vals[:-1],  # [T+1] --> [T]
        actions[:-1],  # [T+1] --> [T]
        target_preds.q_vals[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[1:],  # [T+1] --> [T]
        discounts[1:],
        is_last[1:],
        lambda_[1:],
        )  # [T+1] --> [T]

        # ensure target = 0 when episode terminates
        target_q_t = target_q_t * non_terminal[:-1]
        batch_td_error = target_q_t - q_t
        batch_td_error = batch_td_error * loss_mask[:-1]


        # 1. Prepare Inputs (similar to original)
        rewards = make_float(rewards)
        rewards = rewards - config["STEP_COST"]      # [T+1, B]
        is_last = make_float(is_last)
        discounts = non_terminal * config["DISCOUNT"] # [T+1, B]
        lambda_ = config["TD_LAMBDA"]

        # 2. Align Time Steps for rlax.q_lambda
        q_tm1 = online_preds.q_vals[:-1]      # Online Q(s_t, a) [T, B, A]
        a_tm1 = actions[:-1]                  # Action a_t [T, B]
        r_t = rewards[1:]                     # Reward r_{t+1} [T, B]
        discount_t = discounts[1:]            # Discount gamma_{t+1} [T, B]
        q_t_target = target_preds.q_vals[1:]  # Target Q(s_{t+1}, a') [T, B, A]

        is_last = is_last[1:]                 # Is last t+1 [T, B]
        loss_mask = loss_mask[:-1]            # Valid transitions mask [T, B]
        non_terminal = non_terminal[1:]       # Non-terminal mask [T, B]
        

        # 3. Calculate TD Error
        # We map over the batch dimension (axis 1)
        # We vmap it to handle the batch dimension [T, B, ...] -> [T, B] output
        # Get N-step transformed TD error and loss.
        batch_td_error_fn = jax.vmap(losses.q_learning_lambda_td, in_axes=1, out_axes=1)
        q_t, target_q_t = batch_td_error_fn(
            q_tm1,        # [T, B, A] -> processed as B sequences of [T, A]
            a_tm1,        # [T, B]   -> processed as B sequences of [T]
            q_t_target,   # [T, B, A] -> processed as B sequences of [T, A]
            r_t,          # [T, B]   -> processed as B sequences of [T]
            discount_t,   # [T, B]   -> processed as B sequences of [T]
            is_last,      # [T, B]   -> processed as B sequences of [T]
            lambda_,      # Scalar
        ) # Output shape: [T, B]

        # 4. Apply Mask
        # Zero out errors for invalid transitions
        target_q_t = target_q_t * non_terminal
        batch_td_error = target_q_t - q_t
        batch_td_error = batch_td_error * loss_mask

        # 5. Calculate Loss (Squared Error)
        batch_loss = 0.5 * jnp.square(batch_td_error) # [T, B]

        # 6. Calculate Mean Loss per Batch Item
        # Sum loss over time for each batch item, divide by num valid steps
        batch_loss_mean = batch_loss.sum(0) / loss_mask.sum(0) # [B]

        # 7. Calculate Metrics (similar to original)
        metrics = {
        "0.q_loss": batch_loss.mean(),
        "0.q_td": jnp.abs(batch_td_error).mean(),
        "1.reward": r_t.mean(),
        "z.q_mean": online_preds.q_vals.mean(),
        "z.q_var": online_preds.q_vals.var(),
        }

        log_info = {
        "timesteps": timestep,
        "actions": actions,
        "td_errors": batch_td_error,  # [T]
        "loss_mask": loss_mask,  # [T]
        "q_values": online_preds.q_vals,  # [T, B]
        "q_loss": batch_loss,  # [T, B]
        "q_target": target_q_t,
        }

        return batch_td_error, batch_loss_mean, metrics, log_info

    def total_loss(self, online_params, target_params, batch: Transition, init_state: jax.Array, rng: jax.Array) -> tuple[jax.Array, dict]:
        # Input: batch is sequence [B, T, ...], initial states are [B, ...]

        # --- 1. Online Loss Component ---
        # Unroll online & target Q-networks on the REAL batch data
        # final_online_state: [B, ...]
        # online_preds_real: [B, T, ...]
        # Swap time and batch dimensions
        batch = jax.tree.map(lambda x: x.swapaxes(0, 1), batch) # [T, B, ...]

        # Unpack batch data
        actions = batch.action # [T, B]
        timestep = batch.timestep # [T, B, ...]
        rewards = timestep.reward # [T, B]
        non_terminal = timestep.discount # [T, B]
        is_last = timestep.last() # [T, B]
        loss_mask = is_truncated(timestep) # [T, B]
        obs = timestep.observation # [T, B, ...]
    
        # Unroll online & target Q-networks on the REAL batch data
        resets = 1.0 - non_terminal # [T, B]
        xs = (obs, resets)
        final_online_state, online_preds = self.agent.apply(online_params, init_state, xs)
        final_target_state, target_preds = self.agent.apply(target_params, init_state, xs)

        # Calculate TD-Lambda loss L_online based on online_preds, target_preds, batch.action, batch.timestep.reward, batch.timestep.discount
        all_metrics = {}
        all_log_info = {
        "n_updates": steps,
        }

        T, B = loss_mask.shape[:2]

        td_error, batch_loss, metrics, log_info = self.batch_loss(
            timestep=timestep,
            online_preds=online_preds,
            target_preds=target_preds,
            actions=actions,
            rewards=rewards,
            non_terminal=non_terminal,
            is_last=is_last,
            loss_mask=loss_mask,
        )

        # update L_online
        L_online = batch_loss

        # update metrics
        all_metrics.update({f"{k}/online": v for k, v in metrics.items()})
        all_log_info["online"] = log_info

        # zero pad TD error
        td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
        td_error = jnp.abs(td_error)


        # --- 2. Dyna Loss Component ---
        L_dyna = 0.0
        if self.config["DYNA_COEFF"] > 0:
            # TODO: Implement Dyna Loss Calculation (potentially in a helper method)
            #   - Select starting points (s_t, h_t) from the real batch sequence (e.g., via windows or sampling)
            #   - For each starting point:
            #       - Call simulate_rollout(world_model, model_params, q_network, online_params, s_t, h_t, ...) -> simulated_trajectory
            #       - Combine real_prefix + simulated_trajectory -> combined_trajectory
            #       - Unroll online & target Q-networks on combined_trajectory (using h from start of real segment) -> Q_comb_on, Q_comb_tar
            #       - Calculate TD-Lambda loss L_sim on combined_trajectory (masking appropriately)
            #   - Average L_sim across starting points/simulations -> L_dyna
            # will use time-step + previous rnn-state to simulate
            # next state at each time-step and compute predictions

            # example code from Wilka
            remove_last = lambda x: jax.tree.map(lambda y: y[:-1], x)
            h_tm1_online = concat_first_rest(init_state, remove_last(online_preds.state))
            h_tm1_target = concat_first_rest(init_state, remove_last(target_preds.state))
            x_t = timestep

            dyna_loss_fn = functools.partial(
                self.dyna_loss_fn, params=params, target_params=target_params
            )

            # vmap over batch
            dyna_loss_fn = jax.vmap(dyna_loss_fn, (1, 1, 1, 1, 1, 0), 0)
            _, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
                x_t,
                data.action,
                h_tm1_online,
                h_tm1_target,
                loss_mask,
                jax.random.split(key_grad, B),
            )
            L_dyna += self.dyna_coeff * dyna_batch_loss

            # update metrics with dyna metrics
            all_metrics.update({f"{k}/dyna": v for k, v in dyna_metrics.items()})

            all_log_info["dyna"] = dyna_log_info

        # --- 3. Combine Losses ---
        # TODO: Add importance sampling weights if using prioritized replay
        total_loss = self.config["ONLINE_COEFF"] * jnp.mean(L_online) + self.config["DYNA_COEFF"] * L_dyna

        # --- 4. Prepare Aux Data ---
        # TODO: Package metrics (L_online, L_dyna, etc.)
        # TODO: Package priorities (based on TD_online)
        aux_data = {"metrics": {...}, "priorities": ...}

        return total_loss, aux_data
    
    # TODO: Modify dyna loss fn for my agent class
    def dyna_loss_fn(
        self,
        timesteps: TimeStep,
        actions: jax.Array,
        h_online: jax.Array,
        h_target: jax.Array,
        loss_mask: jax.Array,
        rng: jax.random.PRNGKey,
        params,
        target_params,
    ):
        """

        Algorithm:
        -----------

        Args:
            x_t (TimeStep): [D], timestep at t
            h_tm1 (jax.Array): [D], rnn-state at t-1
            h_tm1_target (jax.Array): [D], rnn-state at t-1 from target network
        """
        window_size = self.config["WINDOW_SIZE"]
        window_size = min(window_size, len(actions))
        window_size = max(window_size, 1)
        roll = partial(rolling_window, size=window_size)
        simulate = partial(
            simulate_n_trajectories,
            network=self.agent,
            params=params,
            num_steps=self.config["SIMULATION_LENGTH"],
            num_simulations=self.config["NUM_SIMULATIONS"],
            policy_fn=self.simulation_policy,
        )

        # first do a rollowing window
        # T' = T-window_size+1
        # K = window_size
        # [T, ...] --> [T', K, ...]
        actions = jax.tree.map(roll, actions)
        timesteps = jax.tree.map(roll, timesteps)
        h_online = jax.tree.map(roll, h_online)
        h_target = jax.tree.map(roll, h_target)
        loss_mask = jax.tree.map(roll, loss_mask)

        def dyna_loss_fn_(t, a, h_on, h_tar, l_mask, key):
            """
            Args:
                t (jax.Array): [window_size, ...]
                h_on (jax.Array): [window_size, ...]
                h_tar (jax.Array): [window_size, ...]
                key (jax.random.PRNGKey): [2]
            """
            # get simulations starting from final timestep in window
            key, key_ = jax.random.split(key)
            # [sim_length, num_sim, ...]
            next_t, sim_outputs_t = simulate(
                h_tm1=jax.tree.map(lambda x: x[-1], h_on),
                x_t=jax.tree.map(lambda x: x[-1], t),
                rng=key_,
            )
            if self.backtracking:
                # we replace last, because last action from data
                # is different than action from simulation
                # [window_size + sim_length, num_sims, ...]
                all_but_last = lambda y: jax.tree.map(lambda x: x[:-1], y)
                all_t = concat_start_sims(all_but_last(t), next_t)
                all_a = concat_start_sims(all_but_last(a), sim_outputs_t.actions)
                # start at beginning of experience data
                start_index = 0
            else:
                all_t = next_t
                all_a = sim_outputs_t.actions
                # start at last timestep before simulation
                start_index = -1

            # NOTE: we're recomputing RNN but easier to read this way...
            # TODO: reuse RNN online param computations for speed (probably not worth it)
            key, key_ = jax.random.split(key)
            h_htm1 = jax.tree.map(lambda x: x[start_index], h_on)
            h_htm1 = repeat(h_htm1, self.num_simulations)
            online_preds = apply_rnn_and_q(
                h_tm1=h_htm1,
                timesteps=all_t,
                task=None,
                rng=key_,
                network=self.network,
                params=params,
                q_fn=self.network.reg_q_fn,
            )

            key, key_ = jax.random.split(key)
            h_htm1 = jax.tree.map(lambda x: x[start_index], h_tar)
            h_htm1 = repeat(h_htm1, self.num_simulations)
            target_preds = apply_rnn_and_q(
                h_tm1=h_htm1,
                timesteps=all_t,
                task=None,
                rng=key_,
                network=self.network,
                params=target_params,
                q_fn=self.network.reg_q_fn,
            )

            all_t_mask = simulation_finished_mask(l_mask, next_t)
            if not self.backtracking:
                all_t_mask = all_t_mask[-self.simulation_length-1:]

            batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
                timestep=all_t,
                online_preds=online_preds,
                target_preds=target_preds,
                actions=all_a,
                rewards=all_t.reward / MAX_REWARD,
                is_last=make_float(all_t.last()),
                non_terminal=all_t.discount,
                loss_mask=all_t_mask,
            )
            return batch_td_error, batch_loss_mean, metrics, log_info

        # vmap over individual windows
        # TD ERROR: [window_size, T + sim_length, num_sim]
        # Loss: [window_size, num_sim, ...]
        if self.combine_real_sim:
            batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(dyna_loss_fn_)(
                timesteps,  # [T, W, ...]
                actions,  # [T, W]
                h_online,  # [T, W, D]
                h_target,  # [T, W, D]
                loss_mask,  # [T, W]
                jax.random.split(rng, len(actions)),  # [W, 2]
            )
        else:
            batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(
                dyna_loss_fn_, (1, 1, 1, 1, 1, 0), 0
            )(
                timesteps,  # [T, W, ...]
                actions,  # [T, W]
                h_online,  # [T, W, D]
                h_target,  # [T, W, D]
                loss_mask,  # [T, W]
                jax.random.split(rng, window_size),  # [W, 2]
            )

        # figuring out how to incorporate windows into TD error is annoying so punting
        # TODO: incorporate windowed overlappping TDs into TD error
        batch_td_error = batch_td_error.mean()  # [num_sim]
        batch_loss_mean = batch_loss_mean.mean()  # []

        return batch_td_error, batch_loss_mean, metrics, log_info

# --- Simulation policy ---
# One greedy, others epsilon-greedy
vals = np.logspace(num=256, start=1, stop=3, base=0.1)
epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
epsilons = jnp.concatenate((jnp.array((0,)), epsilons))

def simulation_policy(preds: Predictions, sim_rng: jax.Array):
    q_values = preds.q_vals
    assert q_values.shape[0] == epsilons.shape[0]
    sim_rng = jax.random.split(sim_rng, q_values.shape[0])
    return jax.vmap(base_agent.epsilon_greedy_act, in_axes=(0, 0, 0))(
        q_values, epsilons, sim_rng
    )

# --- Helper Functions ---

def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  rng: jax.random.PRNGKey,
  agent: nn.Module,
  params: Params,
  policy_fn: SimPolicy = None,
  num_steps: int = 5,
  num_simulations: int = 5,
):
    """

    return predictions and actions for every time-step including the current one.

    This first applies the model to the current time-step and then simulates T more time-steps.
    Output is num_steps+1.

    Args:
        x_t (TimeStep): [D]
        h_tm1 (RnnState): [D]
        rng (jax.random.PRNGKey): _description_
        agent (nn.Module): _description_
        params (Params): _description_
        num_steps (int, optional): _description_. Defaults to 5.
        num_simulations (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """

    def initial_predictions(x, h_tm1):
        # roll through RNN
        # Format inputs for apply fn
        obs, discount = x.observation, x.discount
        obs = obs[np.newaxis, :]
        discount = discount[np.newaxis, :]
        resets = 1.0 - discount
        x = (obs, resets)
        h_t, preds = agent.apply(params, h_tm1, x)
        
        # remove time dim
        h_t = jax.tree.map(lambda h: h.squeeze(0), h_t)
        preds = jax.tree.map(lambda p: p.squeeze(0), preds)

        return x, h_t, preds

    # by giving state as input and returning, will
    # return copies. 1 for each sampled action.
    rng, rng_ = jax.random.split(rng)

    # one for each simulation
    # [N, ...]
    # replace (x_t, task) with N-copies
    x_t = jax.tree_map(lambda x: jnp.repeat(x[None], num_simulations, axis=0), x_t)
    h_tm1 = jax.tree_map(lambda x: jnp.repeat(x[None], num_simulations, axis=0), h_tm1)
    x_t, h_t, preds_t = jax.vmap(initial_predictions, in_axes=(0, 0))(x_t, h_tm1)

    # choose epsilon-greedy action
    a_t = policy_fn(preds_t, rng_)

    def _single_model_step(carry, unused):
        (timestep, agent_state, a, rng) = carry

        ###########################
        # 1. use state + action to predict next state
        ###########################
        rng, rng_ = jax.random.split(rng)

        # apply world model to get next timestep
        next_timestep = agent.apply_world_model(timestep.state, a, rng_)

        # Format inputs for apply fn
        obs, discount = next_timestep.observation, next_timestep.discount
        obs = obs[np.newaxis, :]
        discount = discount[np.newaxis, :]
        resets = 1.0 - discount
        x = (obs, resets)

        # get next agent state and actions
        next_agent_state, next_preds = agent.apply(params, agent_state, x)

        # remove time dim
        next_agent_state = jax.tree.map(lambda x: x.squeeze(0), next_agent_state)
        next_preds = jax.tree.map(lambda x: x.squeeze(0), next_preds)

        # get next actions
        next_a = policy_fn(next_preds, rng_)

        # format outputs
        carry = (next_timestep, next_agent_state, next_a, rng)
        sim_output = SimulationOutput(
            predictions=next_preds,
            actions=next_a,
        )
        return carry, (next_timestep, sim_output)

    ################
    # get simulation outputs
    ################
    initial_carry = (x_t, h_t, a_t, rng)
    _, (next_timesteps, sim_outputs) = jax.lax.scan(
        f=_single_model_step, init=initial_carry, xs=None, length=num_steps
    )

    # sim_outputs.predictions: [T, N, ...]
    # concat [1, ...] with [N, T, ...]
    sim_outputs = SimulationOutput(
        predictions=concat_first_rest(preds_t, sim_outputs.predictions),
        actions=concat_first_rest(a_t, sim_outputs.actions),
    )
    all_timesteps = concat_first_rest(x_t, next_timesteps)
    return all_timesteps, sim_outputs

def simulate_rollout(world_model: WorldModel, model_params, model_state: WorldModelState,
                     q_network: RecurrentQNetwork, q_params,
                     initial_sim_state_info: WorldModel.StateInfo, initial_rnn_state: AgentState,
                     sim_length: int, rng: jax.Array, simulation_policy_fn: callable) -> Transition:
    # Uses jax.lax.scan to perform the rollout
    # Inputs: world model, q-network (for policy), initial states, length, policy
    # Output: A Transition object containing sequences of (sim_obs, sim_reward, sim_done, sim_action, sim_rnn_state)
    def scan_step(carry, _):
        # TODO: Unpack carry (current_sim_state_info, current_rnn_state, current_model_state, rng)
        # TODO: Get Q-values using q_network(current_rnn_state, current_sim_state_info.observation) (requires adapting __call__ or timestep creation)
        # TODO: Choose action using simulation_policy_fn(q_vals, rng)
        # TODO: Predict next step using world_model.apply_model(...) -> next_wm_output, next_model_state
        # TODO: Update RNN state using q_network single step logic based on next_wm_output.next_observation -> next_rnn_state
        # TODO: Prepare next_sim_state_info
        # TODO: Assemble output tuple (sim_obs, sim_reward, sim_done, sim_action, sim_rnn_state)
        # TODO: Return next carry and output tuple
        pass
    # TODO: Define initial carry
    # TODO: Call jax.lax.scan(scan_step, initial_carry, None, length=sim_length)
    # TODO: Package scan outputs into a Transition object (or similar structure)
    pass

def simulation_policy_fn(q_vals, rng, epsilon):
    # Example: Epsilon-greedy
    # TODO: Implement epsilon-greedy logic based on q_vals and epsilon
    pass

# --- Training Loop Structure (Conceptual) ---

def make_train(config, env, env_params):
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), state, action, env_params)

    def train(rng):
        # Initialize environment
        rng, _rng = jax.random.split(rng, 2)
        init_timestep = vmap_reset(config["N_ENVS"])(_rng)

        # Initialize DynaAgent
        agent = DynaAgent(
            config=config
        )
        rng, init_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_agent_state = DynaAgent.initialize_carry(config["NUM_ENVS"], config["RNN_HIDDEN_DIM"])
        online_params = agent.init(init_rng, init_agent_state, init_x)
        target_params = online_params
        
        # Initialize Optimizer
        optimizer = optax.adam(learning_rate=config["LEARNING_RATE"])
        opt_state = optimizer.init(online_params)
        
        # Initialize CustomTrainState
        train_state = CustomTrainState(
            params=online_params,
            target_network_params=target_params,
            opt_state=opt_state,
            timesteps=0,
            n_updates=0
        )
        
        # Initialize Trajectory Replay Buffer
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"],
            min_length_time_axis=config["MIN_BUFFER_SIZE"],
            sample_batch_size=config["N_ENVS"],
            add_batch_size=config["N_ENVS"],
            sample_sequence_length=config["SEQUENCE_LENGTH"],
            period=config["PERIOD"],
        )

        dummy_timestep = TimeStep(
            observation=jnp.zeros((env.observation_space(env_params).shape[0],)),
            reward=jnp.zeros((1,)),
            done=jnp.zeros((1,)),
            state=None,
        )
        dummy_agent_state = jnp.zeros((config["RNN_HIDDEN_DIM"],))
        dummy_transition = Transition(
            timestep=dummy_timestep,
            action=jnp.array(0),
            agent_state=dummy_agent_state
        )
        buffer_state = buffer.init(dummy_transition)
        
        # Define actor_step function
        def actor_step(carry, unused):
            agent_state, timestep, rng = carry
            rng, _rng = jax.random.split(rng)

            # Format inputs for apply fn
            obs, discount = timestep.observation, timestep.discount
            obs = obs[np.newaxis, :]
            discount = discount[np.newaxis, :]
            resets = 1.0 - discount
            x = (obs, resets)

            # Get action from agent
            next_agent_state, preds = agent.apply(
                train_state.online_params,
                agent_state,
                x,
            )
            q_vals = preds.q_vals

            # TODO: change to sim policy
            action = jnp.argmax(q_vals, axis=-1)

            # remove time dim
            action = action.squeeze(axis=0)

            # Get next timestep
            next_timestep = vmap_step(config["N_ENVS"])(_rng, timestep.state, action)

            # Create transition
            transition = Transition(timestep=next_timestep, action=action, agent_state=next_agent_state)

            return (next_agent_state, next_timestep, rng), transition
        
        # Define simulation_policy function
        def simulation_policy(q_vals, rng):
            return jnp.argmax(q_vals)
        
        # Initialize DynaLossFn instance
        loss_fn = DynaLossFn(
            q_network=agent,
            config=config,
            simulation_policy_fn=simulation_policy
        )

        def _train_step(runner_state, unused):
            train_state, env_timestep, agent_state, buffer_state, rng = runner_state

            # --- 1. Collect Experience ---
            # Use jax.lax.scan with actor_step and env.step (vmap_step) for config["TRAINING_INTERVAL"] steps
            #       - Manage agent_state (RNN state) across steps
            #       - Collect sequence of Transitions (including agent_state) -> traj_batch
            (agent_state, timestep, rng), traj_batch = jax.lax.scan(
                actor_step,
                (agent_state, env_timestep, rng),
                None,
                config["TRAINING_INTERVAL"]
            )

            # --- 2. Add to Buffer ---
            # Transpose traj_batch if needed ([T, B] -> [B, T])
            traj_batch = jax.tree.map(lambda x: jnp.transpose(x, (1, 0)), traj_batch)

            # Add traj_batch to buffer: buffer_state = buffer.add(buffer_state, traj_batch)
            buffer_state = buffer.add(buffer_state, traj_batch)

            # Update total timesteps counter in train_state
            train_state = train_state.replace(timesteps=train_state.timesteps + config["TRAINING_INTERVAL"])

            # --- 3. Learning Update ---
            def _learn_step(train_state, buffer_state, rng):
                # Split RNG for sampling and loss calculation
                rng, sample_rng, loss_rng = jax.random.split(rng, 3)
                
                # Sample batch of sequences from buffer -> sampled_batch
                sampled_batch = buffer.sample(buffer_state, sample_rng)

                # Get initial RNN states (online/target) from sampled_batch.experience.agent_state
                init_state = jax.tree.map(
                    lambda x: x[:, 0],  # Take first element along time dimension
                    sampled_batch.experience.agent_state
                )

                # TODO: Implement Burn-in logic (optional) -> update initial states, get learn_batch

                # Call jax.value_and_grad(loss_fn.calculate_loss, has_aux=True)(...)
                #       - Pass online_params, target_params, model_params, learn_batch, initial_states, rng
                #       - Returns: (loss, (aux_data)), grads
                (loss, aux_data), grads = jax.value_and_grad(loss_fn.calculate_loss, has_aux=True)(
                    train_state.online_params,
                    train_state.target_network_params,
                    sampled_batch,
                    init_state,
                    loss_rng
                )

                # Apply gradients: train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.apply_gradients(grads=grads)

                # TODO: Update priorities in buffer: buffer_state = buffer.set_priorities(buffer_state, indices, aux_data["priorities"])
                # Increment update counter in train_state
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)

                # TODO: Log metrics from aux_data["metrics"]

                return train_state, buffer_state, rng

            # Check learning conditions (timesteps > LEARNING_STARTS, buffer.can_sample)
            # Use jax.lax.cond to call perform_learn_step or do nothing
            # train_state, buffer_state = jax.lax.cond(...)
            train_state, buffer_state, rng = jax.lax.cond(
                (train_state.timesteps > config["LEARNING_STARTS"]) & buffer.can_sample(buffer_state),
                lambda ts, bs, r: _learn_step(ts, bs, r),
                lambda ts, bs, r: (ts, bs, r),
                train_state,
                buffer_state,
                rng
            )

            # --- 4. Update Target Network ---
            # Periodically copy online_params to target_network_params in train_state
            # TODO: Use optax.incremental_update to update target_network_params
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(target_network_params=ts.params),
                lambda ts: ts,
                train_state
            )

            # --- 5. Logging / Evaluation ---
            # TODO: Periodically log learner metrics and run evaluation rollouts

            # --- 6. Return updated states ---
            return (train_state, timestep, agent_state, buffer_state, rng), {}

        # JIT the _train_step function
        _train_step = jax.jit(_train_step)

        # Initialize the initial states
        runner_state = (
            train_state,
            init_timestep,
            init_agent_state,
            buffer_state,
            rng
        )

        # Run the main loop
        runner_state, _ = jax.lax.scan(_train_step, runner_state, None, config["NUM_UPDATES"])

        # Return results
        return runner_state

    return train # End of make_train

# --- Main Execution ---
if __name__ == "__main__":
    # Define config dictionary
    config = {
        # --- Environment Settings ---
        "ENV_NAME": "Craftax-Symbolic-v1", # Example, adjust as needed
        "N_ENVS": 4,  # Number of parallel environments (PureJaxRL DQN used 10, can increase)

        # --- Training Loop Settings ---
        "TOTAL_TIMESTEPS": 5e5,  # Total environment steps (PureJaxRL DQN used 5e5)
        "TRAINING_INTERVAL": 1,   # How many env steps per actor sequence collection (R2D2/ACME often use 1)
                                # NOTE: If 1, buffer adds [B=N_ENVS, T=1]. If >1, adds [B=N_ENVS, T=INTERVAL]
        "LEARNING_STARTS": 10000, # Timesteps before learning begins (PureJaxRL DQN used 10k)
        "TARGET_UPDATE_INTERVAL": 250, # How many LEARNER UPDATES between target network syncs (R2D2 uses ~2500 steps)

        # --- Network Settings ---
        "RNN_HIDDEN_DIM": 256,     # Size of RNN hidden state (Dyna code used 256)
        "MLP_HIDDEN_DIM": 128,     # Hidden dim for observation encoder MLP
        "NUM_MLP_LAYERS": 1,       # Layers for observation encoder MLP
        "Q_HIDDEN_DIM": 512,       # Hidden dim for Q-head MLP (Dyna code used 512)
        "NUM_Q_LAYERS": 1,         # Layers for Q-head MLP (Dyna code used 1)
        "ACTIVATION": "relu",      # Activation function
        "USE_BIAS": True,          # Whether to use bias in Dense layers

        # --- Optimizer Settings ---
        "LEARNING_RATE": 2.5e-4,   # Learning rate (PureJaxRL DQN used 2.5e-4)
        "LR_LINEAR_DECAY": False,  # Whether to use linear LR decay
        "EPS_ADAM": 1e-5,          # Adam optimizer epsilon (ACME default 1e-5)
        "MAX_GRAD_NORM": 40.0,     # Gradient clipping norm (ACME default 40.0)

        # --- Buffer Settings ---
        "BUFFER_SIZE": 50000,      # Total transitions in buffer (R2D2 often uses 1M+, adjust based on memory)
        "BUFFER_BATCH_SIZE": 64,   # Batch size sampled from buffer for learning (e.g., 32, 64)
        "SEQUENCE_LENGTH": 40,     # Length of sequences sampled from buffer (R2D2 uses ~80)
        "PERIOD": 1,               # Store sequences overlapping by N-1 steps (1 is standard)
        "BURN_IN_LENGTH": 4,       # Number of steps to burn-in RNN state (0 disables, R2D2 uses ~half sequence length)

        # --- Prioritized Replay Settings ---
        "PRIORITY_EXPONENT": 0.9,            # Alpha exponent for PER (fbx default 0.9)
        "IMPORTANCE_SAMPLING_EXPONENT": 0.6, # Beta exponent for PER IS weights (DynaLossFn default 0.6)
        "MAX_PRIORITY_WEIGHT": 0.9,          # Mixture coefficient for max/mean priority (DynaLossFn default 0.9)

        # --- Loss Function Settings ---
        "GAMMA": 0.99,             # Discount factor
        "TD_LAMBDA": 0.9,         # TD-Lambda parameter (R2D2 uses 0.95)
        "STEP_COST": 0.001,          # Optional cost added per step (DynaLossFn default 0.0)
        "ONLINE_COEFF": 1.0,       # Weight for the loss on real data
        "DYNA_COEFF": 1.0,         # Weight for the loss on simulated data (DynaLossFn default 1.0)

        # --- Dyna Simulation Settings ---
        "NUM_SIMULATIONS": 2,      # Number of parallel simulations per starting state (DynaLossFn default 2)
        "SIMULATION_LENGTH": 5,    # Length of each simulated rollout (DynaLossFn default 5)
        "WINDOW_SIZE": 20,         # Size of the rolling window on real data to trigger simulations (DynaLossFn default 20)
        "SIM_EPSILON_SETTING": 1,  # How simulation epsilon is chosen (1: ACME logspace, 2: 0.9-0.1 logspace, 3: fixed)
        "SIM_EPSILON": 0.1,        # Epsilon value if SIM_EPSILON_SETTING is 3

        # --- Actor Settings (Exploration) ---
        # Choose one exploration strategy
        "FIXED_EPSILON": 0,        # 0: Linear Decay, 1: Fixed Logspace (ACME), 2: Fixed Logspace (0.9-0.1)
        "EPSILON_START": 1.0,      # Starting epsilon for linear decay
        "EPSILON_FINISH": 0.05,    # Final epsilon for linear decay
        "EPSILON_ANNEAL_TIME": 25e4,# Timesteps to anneal epsilon over (PureJaxRL DQN used 25e4)

        # --- Miscellaneous ---
        "SEED": 42,
        "LEARNER_LOG_PERIOD": 100,  # How many LEARNER UPDATES between logging losses/metrics
        "EVAL_LOG_PERIOD": 100,     # How many LEARNER UPDATES between running evaluation episodes
        "GRADIENT_LOG_PERIOD": 500, # How many LEARNER UPDATES between logging gradients (0 to disable)
    }

    # Create env and env_params
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env = TimestepWrapper(env)
    env_params = env.default_params

    # Call make_train(config, env, env_params)
    train = make_train(config, env, env_params)
    train(jax.random.PRNGKey(config["SEED"]))