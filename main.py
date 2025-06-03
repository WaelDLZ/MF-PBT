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

    # Handle default example case
    if args.env is None and args.hpo is None:
        return _load_example_config()

    # Validate arguments
    _validate_arguments(args)

    # Load configuration file
    config_file = _get_config_file_path(args.env, args.hpo)
    config = _load_and_update_config(config_file, args)

    # Handle variance-exploitation mode
    if args.variance_exploitation:
        config = _apply_variance_exploitation_mode(config, args.hpo)

    # Set default experiment name if not provided
    if args.exp_name is None:
        config["exp_name"] = _generate_experiment_name(args, config)

    return config, args.hpo


def _create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="MF-PBT Hyperparameter Optimization")

    # Required arguments
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


def _generate_experiment_name(args, config: dict) -> str:
    """Generate default experiment name."""
    suffix = "_variance_exploit" if args.variance_exploitation else ""
    return f"{args.hpo}/{args.env}/n_agents_{config['num_agents']}{suffix}"


def _load_example_config():
    """Load example configuration for default case."""
    env_choice = "inverted_pendulum"
    hpo_choice = "mfpbt"
    config_file = os.path.join(PATH_TO_MAIN_PROJECT, "configurations/example.yml")

    config = yaml.safe_load(open(config_file))
    config["exp_name"] = f"{hpo_choice}/{env_choice}/n_agents_{config['num_agents']}"

    return config, hpo_choice


def _get_config_file_path(env_choice: str, hpo_choice: str) -> str:
    """Get the configuration file path based on environment and HPO algorithm."""
    config_algo = "random_search" if hpo_choice == "random_search" else hpo_choice

    return os.path.join(
        PATH_TO_MAIN_PROJECT, f"configurations/{env_choice}/{config_algo}.yml"
    )


def _load_and_update_config(config_file: str, args) -> dict:
    """Load configuration from file and update with command line arguments."""
    config = yaml.safe_load(open(config_file))

    # Update them if needed
    for arg in args._get_kwargs():
        (key, value) = arg
        # Handle argument name mapping
        if key == "logging_directory":
            key = "logging_directory"
        if value is not None and key in config.keys():
            config[key] = value

    return config


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
