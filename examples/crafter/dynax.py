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
# Assume gymnax environment imports

# --- Dataclasses ---

# for single env, single timestep
@struct.dataclass
class TimeStep:
    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    env_state: struct.PyTreeNode # Raw environment state

@struct.dataclass
class AgentState:
    # Represents the recurrent state of the agent (e.g., LSTM state)
    rnn_state: struct.PyTreeNode # Could be a tuple (h, c) or just h

@struct.dataclass
class Transition:
    # Structure stored in the replay buffer
    timestep: TimeStep      # Information at time t (s_t, r_t, done_t)
    action: jax.Array       # Action taken at time t (a_t)
    agent_state: AgentState # Agent's RNN state *before* processing timestep t (h_{t-1})

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

@struct.dataclass
class RunnerState():
    train_state: CustomTrainState  # Contains network params, optimizer state, etc.
    env_timestep: TimeStep         # Current environment state
    agent_state: AgentState        # Current RNN state of the agent
    buffer_state: Any              # State of the replay buffer
    rng: jax.Array                # Random number generator key

@struct.dataclass
class Predictions:
    q_vals: jax.Array
    # Optionally store other network outputs if needed

@struct.dataclass
class WorldModelState:
    # Placeholder for potential learned world model state
    internal_state: Optional[jax.Array] = None

@struct.dataclass
class WorldModelOutput:
    next_observation: jax.Array
    reward: jax.Array
    done: jax.Array # Predicted termination
    # Potentially next env_state if model predicts it directly
    next_env_state: Optional[struct.PyTreeNode] = None


# --- Network Definition ---

class RecurrentQNetwork(nn.Module):
    action_dim: int
    rnn_hidden_dim: int
    # Other config (MLP dims, activation, etc.)

    def setup(self):
        # TODO: Define observation_encoder (e.g., MLP)
        # TODO: Define rnn_cell (e.g., nn.LSTMCell)
        # TODO: Define q_head (e.g., MLP or DuellingMLP)
        pass

    def initialize_carry(self, batch_dims):
        # TODO: Return initial RNN state (zeros) with correct batch shape
        pass

    def __call__(self, agent_state: AgentState, timestep: TimeStep, rng: jax.Array) -> tuple[Predictions, AgentState]:
        # Single step logic (for actor and simulation)
        # TODO: Encode observation from timestep.observation
        # TODO: Pass (agent_state.rnn_state, encoded_obs, timestep.done) through rnn_cell
        # TODO: Pass RNN output through q_head
        # TODO: Return Predictions(q_vals=...) and next AgentState(rnn_state=...)
        pass

    def unroll(self, initial_agent_state: AgentState, sequence: Transition, rng: jax.Array) -> tuple[Predictions, AgentState]:
        # Process a sequence of transitions using jax.lax.scan
        # Input: initial_agent_state (h_0), sequence (timesteps t_1..t_N)
        # Output: Predictions (q_vals q_1..q_N), final_agent_state (h_N)
        # TODO: Implement scan loop calling the single-step logic
        pass

# --- ScannedRNN --- from PureJaxRl
class ScannedRNN(nn.Module):
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
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )

# --- World Model Definition ---

class WorldModel(nn.Module):
    # Base class/interface (can be a simple struct if no learnable params yet)

    @struct.dataclass
    class StateInfo:
        # What the world model needs to predict the next step
        observation: jax.Array
        env_state: struct.PyTreeNode # Needed for GroundTruth model

    def apply_model(self, params, model_state: WorldModelState, current_state_info: StateInfo, action: jax.Array, rng: jax.Array) -> tuple[WorldModelOutput, WorldModelState]:
        # Abstract method
        raise NotImplementedError

class GroundTruthWorldModel(WorldModel):
    env: object # Gymnax environment instance
    env_params: object # Gymnax env params

    def setup(self):
        # No learnable parameters needed
        pass

    def apply_model(self, params, model_state: WorldModelState, current_state_info: WorldModel.StateInfo, action: jax.Array, rng: jax.Array) -> tuple[WorldModelOutput, WorldModelState]:
        # Uses the real environment step function
        # TODO: Call self.env.step(rng, current_state_info.env_state, action, self.env_params)
        # TODO: Extract next_obs, reward, done, next_env_state from the result
        # TODO: Package into WorldModelOutput
        # TODO: Return WorldModelOutput and unchanged model_state (since it's stateless)
        pass

# --- Loss Function ---

@struct.dataclass
class DynaLossFn:
    q_network: RecurrentQNetwork
    world_model: WorldModel
    config: dict # Containing GAMMA, TD_LAMBDA, ONLINE_COEFF, DYNA_COEFF, SIM_LENGTH, etc.
    simulation_policy_fn: callable # Function: (q_vals, rng) -> action

    def calculate_loss(self, online_params, target_params, model_params, batch: Transition, initial_online_state: AgentState, initial_target_state: AgentState, rng: jax.Array) -> tuple[jax.Array, dict]:
        # Input: batch is sequence [B, T, ...], initial states are [B, ...]

        # --- 1. Online Loss Component ---
        # TODO: Unroll online & target Q-networks on the REAL batch data
        # online_preds_real, _ = self.q_network.apply(online_params, initial_online_state, batch.timestep, rng1, method=self.q_network.unroll)
        # target_preds_real, _ = self.q_network.apply(target_params, initial_target_state, batch.timestep, rng2, method=self.q_network.unroll)
        # TODO: Calculate TD-Lambda loss L_online based on online_preds_real, target_preds_real, batch.action, batch.timestep.reward, batch.timestep.discount
        # TODO: Calculate TD errors TD_online for priority updates

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

        # --- 3. Combine Losses ---
        # TODO: Add importance sampling weights if using prioritized replay
        total_loss = self.config["ONLINE_COEFF"] * jnp.mean(L_online) + self.config["DYNA_COEFF"] * L_dyna

        # --- 4. Prepare Aux Data ---
        # TODO: Package metrics (L_online, L_dyna, etc.)
        # TODO: Package priorities (based on TD_online)
        aux_data = {"metrics": {...}, "priorities": ...}

        return total_loss, aux_data

# --- Helper Functions ---

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
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        # Initialize environment
        rng, _rng = jax.random.split(rng, 2)
        init_obs, env_state = vmap_reset(config["N_ENVS"])(_rng)

        # Initialize RecurrentQNetwork (agent), get initial params
        agent = RecurrentQNetwork(
            action_dim=env.action_space.n,
            rnn_hidden_dim=config["RNN_HIDDEN_DIM"]
        )
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
        dummy_timestep = TimeStep(
            observation=dummy_obs,
            reward=jnp.array(0.0),
            done=jnp.array(0),
            env_state=None
        )
        dummy_agent_state = AgentState(rnn_state=agent.initialize_carry((1,)))
        online_params = agent.init(init_rng, dummy_agent_state, dummy_timestep, rng)
        target_params = online_params
        
        # Initialize GroundTruthWorldModel, get initial (empty) params
        world_model = GroundTruthWorldModel(env=env, env_params=env_params)
        model_params = {}
        
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
        dummy_transition = Transition(
            timestep=dummy_timestep,
            action=jnp.array(0),
            agent_state=dummy_agent_state
        )
        buffer_state = buffer.init(dummy_transition)
        
        # Define actor_step function
        def actor_step(carry, unused):
            agent_state, timestep, rng = carry
            rng, action_rng = jax.random.split(rng)
            predictions, next_agent_state = agent.apply(
                train_state.online_params,
                agent_state,
                timestep,
                action_rng
            )
            action = jnp.argmax(predictions.q_vals, axis=-1)
            obs, env_state, reward, done, _ = vmap_step(config["N_ENVS"])(rng, env_state, action)
            next_timestep = TimeStep(observation=obs, reward=reward, done=done, env_state=env_state)
            transition = Transition(timestep=next_timestep, action=action, agent_state=next_agent_state)

            return (next_agent_state, next_timestep, rng), transition
        
        # Define simulation_policy function
        def simulation_policy(q_vals, rng):
            return jnp.argmax(q_vals)
        
        # Initialize DynaLossFn instance
        loss_fn = DynaLossFn(
            q_network=agent,
            world_model=world_model,
            config=config,
            simulation_policy_fn=simulation_policy
        )
        
        # Initialize RunnerState
        rng, env_rng = jax.random.split(rng)
        initial_timestep = env.reset(env_rng, env_params)
        initial_agent_state = AgentState(rnn_state=agent.initialize_carry((1,)))
        runner_state = RunnerState(
            train_state=train_state,
            env_timestep=initial_timestep,
            agent_state=initial_agent_state,
            buffer_state=buffer_state,
            rng=rng
        )

        def _train_step(runner_state, _):
            # --- 1. Collect Experience ---
            # Use jax.lax.scan with actor_step and env.step (vmap_step) for config["TRAINING_INTERVAL"] steps
            #       - Manage agent_state (RNN state) across steps
            #       - Collect sequence of Transitions (including agent_state) -> traj_batch
            (agent_state, timestep, rng), traj_batch = jax.lax.scan(
                actor_step,
                (runner_state.agent_state, runner_state.env_timestep, runner_state.rng),
                None,
                config["TRAINING_INTERVAL"]
            )

            # --- 2. Add to Buffer ---
            # Transpose traj_batch if needed ([T, B] -> [B, T])
            traj_batch = jax.tree.map(lambda x: jnp.transpose(x, (1, 0)), traj_batch)

            # Add traj_batch to buffer: buffer_state = buffer.add(buffer_state, traj_batch)
            buffer_state = buffer.add(buffer_state, traj_batch)

            # Update total timesteps counter in train_state
            train_state = runner_state.train_state
            train_state = train_state.replace(timesteps=train_state.timesteps + config["TRAINING_INTERVAL"])

            # Update runner_state (new env_timestep, agent_state, rng)
            runner_state = RunnerState(
                train_state=train_state,
                env_timestep=timestep,
                agent_state=agent_state,
                buffer_state=buffer_state,
                rng=rng
            )

            # --- 3. Learning Update ---
            model_params = {} # Empty for GroundTruthWorldModel

            def _learn_step(train_state, buffer_state, rng):
                # TODO: Sample batch of sequences from buffer -> sampled_batch
                # TODO: Get initial RNN states (online/target) from sampled_batch.experience.agent_state
                # TODO: Implement Burn-in logic (optional) -> update initial states, get learn_batch
                # TODO: Call jax.value_and_grad(loss_fn.calculate_loss, has_aux=True)(...)
                #       - Pass online_params, target_params, model_params, learn_batch, initial_states, rng
                #       - Returns: (loss, (aux_data)), grads
                # TODO: Apply gradients: train_state = train_state.apply_gradients(grads=grads)
                # TODO: Update priorities in buffer: buffer_state = buffer.set_priorities(buffer_state, indices, aux_data["priorities"])
                # TODO: Increment update counter in train_state
                # TODO: Log metrics from aux_data["metrics"]
                # return train_state, buffer_state
                pass

            # TODO: Check learning conditions (timesteps > LEARNING_STARTS, buffer.can_sample)
            # TODO: Use jax.lax.cond to call perform_learn_step or do nothing
            # train_state, buffer_state = jax.lax.cond(...)

            # --- 4. Update Target Network ---
            # TODO: Periodically copy online_params to target_network_params in train_state

            # --- 5. Logging / Evaluation ---
            # TODO: Periodically log learner metrics and run evaluation rollouts

            # --- 6. Return updated runner state ---
            # TODO: Update runner_state with new train_state, buffer_state, rng
            # return runner_state, {}
            pass

        # TODO: JIT the _train_step function
        # TODO: Initialize the initial runner_state
        # TODO: Run the main loop: jax.lax.scan(_train_step, initial_runner_state, None, config["NUM_UPDATES"])
        # TODO: Return results

    return train # End of make_train

# --- Main Execution ---
if __name__ == "__main__":
    # TODO: Define config dictionary
    # TODO: Create env and env_params
    # TODO: Call make_train(config, env, env_params)
    # TODO: Run the returned train function
    pass