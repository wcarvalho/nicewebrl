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

@struct.dataclass
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

# --- Network Definition ---
# Based on ScannedRNN from PureJaxRL
class DynaAgent(nn.Module):
    config: dict

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
    
    def apply_world_model(self, env, env_params, env_state: Any, action: jax.Array, rng: jax.Array) -> TimeStep:
        """
        Simulates one step using the 'world model' (ground truth env).
        """
        # vmap the step function
        vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng, n_envs), env_state, action, env_params)
        
        # Implement ground truth env.step call
        next_obs, next_env_state, reward, done, _ = vmap_step(self.config["N_ENVS"])(rng, env_state, action)
        output = TimeStep(
            observation=next_obs,
            reward=reward,
            done=done,
            env_state=next_env_state
        )
        return output

# --- Loss Function ---

@struct.dataclass
class DynaLossFn:
    q_network: DynaAgent
    config: dict # Containing GAMMA, TD_LAMBDA, ONLINE_COEFF, DYNA_COEFF, SIM_LENGTH, etc.
    simulation_policy_fn: callable # Function: (q_vals, rng) -> action

    def calculate_loss(self, online_params, target_params, model_params, batch: Transition, initial_state: AgentState, rng: jax.Array) -> tuple[jax.Array, dict]:
        # Input: batch is sequence [B, T, ...], initial states are [B, ...]

        # --- 1. Online Loss Component ---
        # TODO: Unroll online & target Q-networks on the REAL batch data
        # online_preds_real, _ = self.q_network.apply(online_params, initial_state, batch.timestep, rng1, method=self.q_network.unroll)
        # target_preds_real, _ = self.q_network.apply(target_params, initial_state, batch.timestep, rng2, method=self.q_network.unroll)
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
        agent = DynaAgent(
            config=config
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

            # Get action from agent
            predictions, next_agent_state = agent.apply(
                train_state.online_params,
                agent_state,
                timestep,
                action_rng
            )
            action = jnp.argmax(predictions.q_vals, axis=-1)

            # Get next timestep
            obs, env_state, reward, done, _ = vmap_step(config["N_ENVS"])(rng, env_state, action)
            next_timestep = TimeStep(observation=obs, reward=reward, done=done, env_state=env_state)

            # Create transition
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
        
        # Initialize initial states
        rng, env_rng = jax.random.split(rng)
        initial_timestep = env.reset(env_rng, env_params)
        initial_agent_state = AgentState(rnn_state=agent.initialize_carry((1,)))

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
            model_params = {} # Empty for GroundTruthWorldModel

            def _learn_step(train_state, buffer_state, rng):
                # Split RNG for sampling and loss calculation
                rng, sample_rng, loss_rng = jax.random.split(rng, 3)
                
                # Sample batch of sequences from buffer -> sampled_batch
                sampled_batch = buffer.sample(buffer_state, sample_rng)

                # Get initial RNN states (online/target) from sampled_batch.experience.agent_state
                initial_state = jax.tree.map(
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
                    model_params,
                    sampled_batch,
                    initial_state,
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
            initial_timestep,
            initial_agent_state,
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
        "TD_LAMBDA": 0.95,         # TD-Lambda parameter (R2D2 uses 0.95)
        "STEP_COST": 0.0,          # Optional cost added per step (DynaLossFn default 0.0)
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
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    # Call make_train(config, env, env_params)
    train = make_train(config, env, env_params)
    train(jax.random.PRNGKey(config["SEED"]))