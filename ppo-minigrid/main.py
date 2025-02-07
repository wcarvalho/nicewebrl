import os
import sys
import yaml
import jax
from training.ppo import make_train

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        config: Dictionary containing the training configuration.
    """
    print(f"Loading configuration from {config_path}...", flush=True)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully!")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

def main():
    # Debugging information
    print("=" * 50, flush=True)
    print("STARTING SCRIPT", flush=True)
    print("Python version:", sys.version, flush=True)
    print("Current working directory:", os.getcwd(), flush=True)
    print("=" * 50, flush=True)

    # Load config from yaml
    print("Loading config from default.yaml...")
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    print("NUM_ACTIONS in config:", config["NUM_ACTIONS"])
    
    # Create training function
    train_fn = make_train(config)
    
    # Run training
    rng = jax.random.PRNGKey(0)
    train_fn(rng)

if __name__ == "__main__":
    main()
