from hpo.pbt import PopulationBasedTraining
from hpo.mfpbt import MultipleFrequenciesPopulationBasedTraining
from hpo.ablation import MultipleFrequenciesPopulationBasedTraining as Ablation
from hpo.pb2 import PopulationBasedBandits


from pprint import pprint

import os
import argparse
import yaml

from hpo.pbt import PATH_TO_MAIN_PROJECT


def load_config():
    """Load and validate experiment configuration."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Handle config file loading vs automatic selection
    if args.config_file is not None:
        config = _load_config_from_file(args.config_file)
        hpo_choice = "mfpbt"  # Default, can be overridden by args
    elif args.env is not None and args.hpo is not None:
        config_file = _get_config_file_path(args.env, args.hpo)
        config = _load_config_from_file(config_file)
        hpo_choice = args.hpo
    else:
        return _load_example_config()

    # Validate arguments if env/hpo provided
    if args.env is not None or args.hpo is not None:
        _validate_arguments(args)
        if args.hpo is not None:
            hpo_choice = args.hpo

    # Update config with command line arguments
    config = _update_config_with_args(config, args)

    # Handle variance-exploitation mode
    if args.variance_exploitation:
        config = _apply_variance_exploitation_mode(config, hpo_choice)

    # Set experiment name (either from args or generate default)
    if args.exp_name is not None:
        config["exp_name"] = args.exp_name
    else:
        config["exp_name"] = _generate_experiment_name(args, config, hpo_choice)

    return config, hpo_choice


def _create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="MF-PBT Hyperparameter Optimization")

    # Config file option
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file (overrides env/hpo automatic selection)"
    )

    # Environment and algorithm selection
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--hpo", type=str, help="HPO algorithm")

    # Optional arguments
    parser.add_argument(
        "--frequencies",
        nargs="*",
        type=int,
        help="Frequencies to use, if in MF-PBT, give in the form --frequencies 1 10 20",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name - if we detect an experiment with the same name, we will reload it",
    )
    parser.add_argument(
        "--num-agents", type=int, help="Number of agents to use in parallel"
    )
    parser.add_argument(
        "--num-envs-per-agent",
        type=int,
        help="Number of brax envs to use for each agent",
    )
    parser.add_argument(
        "--num-rounds", type=int, help="Number of rounds to in the experiment"
    )
    parser.add_argument("--bucket-path", type=str, help="path for external logging")
    parser.add_argument(
        "--num-timesteps-round",
        type=int,
        help=(
            "Number of environment steps per round, note that depending on the training config, "
            "batch size and num_minibatches, more steps will be performed"
        ),
    )
    parser.add_argument(
        "--jax-seed",
        type=int,
        default=0,
        help="jax seed used for the training and env simulations",
    )
    parser.add_argument(
        "--numpy-seed",
        type=int,
        default=0,
        help="Numpy seed used for the random initialization of hyperparameters",
    )
    parser.add_argument(
        "--variance-exploitation",
        action="store_true",
        help="Enable variance-exploitation mode: hyperparameters remain fixed, PBT used only for model selection",
    )
    parser.add_argument(
        "--logging-directory", type=str, help="Custom logging directory (defaults to runs/ in project directory)"
    )

    return parser


def _validate_arguments(args):
    """Validate command line arguments."""
    valid_envs = ["ant", "halfcheetah", "hopper", "humanoid", "pusher", "walker2d"]
    valid_hpos = ["pbt", "mfpbt", "ablation", "pb2", "do_nothing", "random_search"]

    if args.env not in valid_envs:
        raise ValueError(
            f"Environment {args.env} not supported. Choose from {valid_envs}"
        )

    if args.hpo not in valid_hpos:
        raise ValueError(
            f"HPO algorithm {args.hpo} not supported. Choose from {valid_hpos}"
        )


def _apply_variance_exploitation_mode(config: dict, hpo_choice: str) -> dict:
    """Apply variance-exploitation mode configuration."""
    print(
        "Variance-exploitation mode enabled: hyperparameters fixed, model selection only"
    )

    if hpo_choice in ["pbt", "pb2", "random_search"]:
        config["routines"] = {
            "explore": None,
            "exploit": {"function_name": "truncation", "function_param": 0.25},
        }
    elif hpo_choice in ["mfpbt", "ablation"]:
        config["routines"] = {"explore": None}

    return config


def _get_config_file_path(env_choice: str, hpo_choice: str) -> str:
    """Get the configuration file path based on environment and HPO algorithm."""
    # Map algorithm choices to config file names
    config_file_map = {
        "pbt": "pbt.yml",
        "mfpbt": "mfpbt.yml", 
        "ablation": "mfpbt.yml",  # Uses same config as mfpbt
        "pb2": "pb2.yml",
        "random_search": "pbt.yml",  # Uses PBT config with no exploration
        "do_nothing": "pbt.yml"
    }
    
    config_filename = config_file_map.get(hpo_choice, "mfpbt.yml")
    
    return os.path.join(
        PATH_TO_MAIN_PROJECT, f"configurations/{env_choice}/{config_filename}"
    )


def _load_config_from_file(config_file: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_file}: {e}")


def _update_config_with_args(config: dict, args) -> dict:
    """Update configuration with command line arguments, handling name mappings."""
    
    # Mapping from argument names to config keys
    arg_to_config_map = {
        "env": "env_name",
        "jax_seed": "jax_seed",
        "numpy_seed": "numpy_seed", 
        "num_agents": "num_agents",
        "num_envs_per_agent": "num_envs_per_agent",
        "num_rounds": "num_rounds",
        "num_timesteps_round": "num_timesteps_round",
        "frequencies": "frequencies",
        "logging_directory": "logging_directory"
    }
    
    for arg_name, value in args._get_kwargs():
        if value is not None:
            # Map argument name to config key
            config_key = arg_to_config_map.get(arg_name, arg_name)
            
            # Only update if the config key exists or is expected
            if config_key in config or arg_name in ["env", "frequencies", "logging_directory"]:
                config[config_key] = value
    
    return config


def _generate_experiment_name(args, config: dict, hpo_choice: str) -> str:
    """Generate default experiment name."""
    env_name = getattr(args, 'env', config.get('env_name', 'unknown'))
    suffix = "_variance_exploit" if args.variance_exploitation else ""
    return f"{hpo_choice}/{env_name}/n_agents_{config['num_agents']}{suffix}"


def _load_example_config():
    """Load example configuration for default case."""
    env_choice = "hopper"
    hpo_choice = "mfpbt"
    config_file = os.path.join(PATH_TO_MAIN_PROJECT, f"configurations/{env_choice}/mfpbt.yml")

    config = _load_config_from_file(config_file)
    config["exp_name"] = f"{hpo_choice}/{env_choice}/n_agents_{config['num_agents']}"

    return config, hpo_choice


if __name__ == "__main__":
    config, hpo_choice = load_config()
    print("Launching experiment with the following config: \n")
    pprint(config)
    print("\n")

    if hpo_choice == "mfpbt":
        alg = MultipleFrequenciesPopulationBasedTraining(**config)
    elif hpo_choice == "pb2":
        alg = PopulationBasedBandits(**config)
    elif hpo_choice == "ablation":
        alg = Ablation(**config)
    elif hpo_choice == "random_search":
        # Random search uses PBT with no exploration/exploitation
        alg = PopulationBasedTraining(**config)
    else:
        alg = PopulationBasedTraining(**config)

    alg.run()
