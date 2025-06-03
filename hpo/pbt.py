import functools
import os
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Dict, List, Any

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

from hpo.utils import exploit_functions, explore_functions
from hpo.utils.logger import Logger
from rl_train.ppo import init_hyperparameters, train_round

PATH_TO_MAIN_PROJECT = str(pathlib.Path(__file__).parent.parent.absolute())


def unmap_data(data):
    return jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data)


def map_data(data: list, num_devices: int, num_agents_per_device: int):
    return jax.tree_map(
        lambda x: x.reshape(num_devices, num_agents_per_device, *x.shape[1:]),
        data,
    )


@dataclass
class Agent:
    index: int
    parent: int
    hyperparameters: dict
    params: dict
    reward: float
    env_steps: int


class PopulationBasedTraining:
    def __init__(
        self,
        num_agents: int,
        exp_name: str,
        env_name: str,
        num_envs_per_agent: int,
        num_timesteps_round: int,
        num_rounds: int,
        hpo_search_space: Dict[str, Any],
        default_hps: Dict[str, Any],
        training_config: Dict[str, Any],
        routines: Dict[str, Any],
        logging_directory: Optional[str] = None,
        backend: str = "positional",
        numpy_seed: int = 0,
        jax_seed: int = 0,
    ) -> None:
        # Initialize core attributes
        self._setup_random_seeds(numpy_seed, jax_seed)
        self._setup_experiment_config(
            num_agents,
            exp_name,
            num_rounds,
            hpo_search_space,
            default_hps,
            training_config,
        )

        # Setup environment and training
        self._setup_environment(
            env_name, backend, num_envs_per_agent, num_timesteps_round
        )
        self._assert_config()

        # Setup HPO routines
        self._setup_hpo_routines(routines, hpo_search_space)

        # Setup JAX and logging
        self._setup_jax_devices(num_agents)
        self._setup_logging(logging_directory, exp_name)

        # Initialize experiment
        self.ranking_indexes = None
        self.starting_round = None
        self._initiate_experiment()

    def _setup_random_seeds(self, numpy_seed: int, jax_seed: int) -> None:
        """Initialize random number generators."""
        self.numpy_rng = np.random.default_rng(numpy_seed)
        self.jax_seed = jax_seed

    def _setup_experiment_config(
        self,
        num_agents: int,
        exp_name: str,
        num_rounds: int,
        hpo_search_space: Dict[str, Any],
        default_hps: Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> None:
        """Setup basic experiment configuration."""
        self.num_agents = num_agents
        self.exp_name = exp_name
        self.num_rounds = num_rounds
        self.hpo_search_space = hpo_search_space
        self.default_hps = default_hps
        self.training_config = training_config

    def _setup_environment(
        self,
        env_name: str,
        backend: str,
        num_envs_per_agent: int,
        num_timesteps_round: int,
    ) -> None:
        """Setup Brax environment configuration."""
        env = envs.get_environment(env_name=env_name, backend=backend)
        self.training_config.update(
            {
                "environment": env,
                "num_timesteps": num_timesteps_round,
                "num_envs": num_envs_per_agent,
            }
        )

    def _setup_hpo_routines(
        self, routines: Dict[str, Any], hpo_search_space: Dict[str, Any]
    ) -> None:
        """Setup hyperparameter optimization routines."""
        # Validation
        if routines["explore"] is not None:
            assert hpo_search_space.keys() == routines["explore"].keys(), (
                "Exploration function and search space must be specified for all hyperparameters"
            )

        # Handle empty search space
        if hpo_search_space is None:
            hpo_search_space = {}

        self.hpo_keys = list(hpo_search_space.keys())
        self.param_keys = [
            "policy_params",
            "value_params",
            "normalizer_params",
            "optimizer_state",
        ]

        # Setup exploration and exploitation functions
        self._explore = {
            key: getattr(explore_functions, routines["explore"][key]["function_name"])(
                **routines["explore"][key]["function_params"],
            )
            for key in self.hpo_keys
        }

        self._exploit = getattr(
            exploit_functions,
            routines["exploit"]["function_name"],
        )(routines["exploit"]["function_param"])

    def _setup_jax_devices(self, num_agents: int) -> None:
        """Setup JAX device configuration."""
        self.num_devices = jax.device_count()
        print(f"Local devices: {self.num_devices}")
        print(f"Devices: {jax.devices()}")

        self.num_agents_per_device = num_agents // self.num_devices
        self._map_data = functools.partial(
            map_data,
            num_devices=self.num_devices,
            num_agents_per_device=self.num_agents_per_device,
        )

    def _setup_logging(self, logging_directory: Optional[str], exp_name: str) -> None:
        """Setup logging path."""
        if logging_directory is not None:
            self.logging_path = os.path.join(logging_directory, exp_name)
        else:
            self.logging_path = os.path.join(PATH_TO_MAIN_PROJECT, "runs", exp_name)

        self.logger = Logger(
            logging_path=self.logging_path,
            hpo_keys=self.hpo_keys,
        )

    def _assert_config(self):
        assert (
            self.training_config["num_minibatches"] * self.training_config["batch_size"]
            >= self.training_config["num_envs"]
        ), "You can't perform training if you have more envs than steps in a batch."

        env_step_per_training_step = (
            self.training_config["batch_size"]
            * self.training_config["unroll_length"]
            * self.training_config["num_minibatches"]
        )

        num_training_steps_per_epoch = np.ceil(
            self.training_config["num_timesteps"] / env_step_per_training_step,
        ).astype(int)

        print(
            f"Note: regarding the arguments given in training config, the agents will perform"
            f" {num_training_steps_per_epoch} training steps at each round. \n"
            f"And one training step requires {env_step_per_training_step} environment steps.\n"
            f"So in total the number of env steps per round will be {num_training_steps_per_epoch * env_step_per_training_step}\n"
            f"And the total experiment will use {self.num_rounds * num_training_steps_per_epoch * env_step_per_training_step} steps per agent.\n",
        )

    def _initiate_experiment(self):
        # This method let us know if we are reloading a previous xp to continue it
        reloading, round_index, log_df, agent_params = self.logger.read_logs()

        if not reloading:
            print(f"New experiment: {self.exp_name}\n")
            self.agents = self._initiate_agents()
            self.starting_round = 0

        else:
            print(f"Reloading experiment {self.exp_name} from round {round_index}\n")
            self.starting_round = round_index
            self.agents = self._initiate_agents()
            self._reload_agents(log_df, agent_params)

    def _initiate_agents(self):
        initial_hps = self._init_hps()
        initial_params = {key: None for key in self.param_keys}

        # Convention at start of experiment agents have no parent
        return [
            Agent(
                index=i,
                parent=-1,
                hyperparameters=initial_hps[i],
                params=initial_params,
                reward=0.0,
                env_steps=0,
            )
            for i in range(self.num_agents)
        ]

    def _init_hps(self):
        # We initiate hps randomly between two values
        # defined in the hpo_search_space dictionary
        values = {key: self._get_init_values(key) for key in self.hpo_keys}

        initial_hps = [
            {key: values[key][i] for key in self.hpo_keys}
            for i in range(self.num_agents)
        ]
        return initial_hps

    def _get_init_values(self, key: str):
        value_min = self.hpo_search_space[key]["init_min"]
        value_max = self.hpo_search_space[key]["init_max"]
        method = self.hpo_search_space[key]["method"]

        if method == "log_uniform":
            # For example if we want to init learning rate uniformly between 1e-5 and 1e-3
            # we expect the user to give exponents
            return np.power(
                10,
                self.numpy_rng.uniform(value_min, value_max, self.num_agents),
            )

        elif method == "uniform":
            return self.numpy_rng.uniform(value_min, value_max, self.num_agents)

        elif method == "log_grid":
            return np.logspace(value_min, value_max, self.num_agents)
        else:
            raise NotImplementedError("I will code other init methods as needed")

    def _reload_agents(self, log_df, agent_params):
        for i in range(len(log_df)):
            line = log_df.iloc[i]
            index = int(line["index"])

            agent = self.agents[index]
            agent.parent = int(line.parent)
            agent.env_steps = int(line.env_steps)
            agent.reward = line.reward
            for hp in self.hpo_keys:
                agent.hyperparameters[hp] = line[hp]
            agent.params = agent_params[index]

        # Then we reperform the evolution step
        self._rank_agents()
        self._perform_exploit()
        self._perform_explore()

    def _format_hyperparameters(self):
        # Get all hyperparameters in the good format
        optimizing_hyperparameters = {}
        for hp in self.hpo_keys:
            values = np.array([agent.hyperparameters[hp] for agent in self.agents])
            optimizing_hyperparameters.update({hp: values})

        hyperparameters = init_hyperparameters(
            hyperparameters=optimizing_hyperparameters,
            default_hyperparameters=self.default_hps,
            num_agents=self.num_agents,
        )
        return hyperparameters

    def _format_params(self, round_index):
        if round_index == 0:
            return {param: None for param in self.param_keys}

        return jax.tree_map(
            lambda *arrays: jnp.stack(arrays),
            *[agent.params for agent in self.agents],
        )

    def _post_process_training(
        self,
        parameters,
        rewards,
        steps_done,
    ):
        for idx in range(self.num_agents):
            self.agents[idx].reward = rewards[idx]
            self.agents[idx].params = jax.tree_map(lambda x: x[idx], parameters)
            self.agents[idx].env_steps += int(steps_done)

    def _rank_agents(self):
        rewards = np.array([agent.reward for agent in self.agents])
        self.ranking_indexes = np.argsort(-rewards)

    def _perform_exploit(self):
        parents = self._exploit(self.ranking_indexes)
        for agent in self.agents:
            agent.parent = parents[agent.index]

    def _perform_explore(self):
        for agent in self.agents:
            if agent.parent != agent.index:
                parent = self.agents[agent.parent]
                # Update hyperparameters: explore
                for key in self.hpo_keys:
                    value = parent.hyperparameters[key]
                    new_value = self._explore[key](value)
                    agent.hyperparameters[key] = new_value
                # Copy params:
                agent.params = deepcopy(parent.params)

    def run(self):
        rng = jax.random.PRNGKey(self.jax_seed)
        key, key_round = jax.random.split(rng)
        key_rounds = jax.random.split(key_round, self.num_agents)
        key_rounds = np.reshape(
            key_rounds,
            (self.num_devices, -1) + key_rounds.shape[1:],
        )
        train_fn = functools.partial(train_round, **self.training_config)
        vmapped_train_fn = jax.vmap(train_fn)
        pmapped_train_fn = jax.pmap(vmapped_train_fn, axis_name="i")

        for round_index in range(self.starting_round, self.num_rounds):
            print(f"Round {round_index} of {self.num_rounds - 1}")
            start = perf_counter()

            # Get hyperparameters of all agents
            hyperparameters = self._format_hyperparameters()
            hyperparameters = self._map_data(hyperparameters)

            # Get params of all agents
            parameters = self._format_params(round_index=round_index)
            parameters = self._map_data(parameters)

            # Perform training
            out = pmapped_train_fn(hyperparameters, parameters, key_rounds)
            _params, _rewards, key_rounds, steps_done = jax.block_until_ready(out)

            end = perf_counter()

            # All agents perform the same number of steps
            steps_done = jnp.mean(steps_done)

            total_sps = self.num_agents * steps_done / (end - start)
            sps = total_sps / self.num_agents

            print(f"Steps per second: total={total_sps:.2f}, per agent = {sps:.2f}")
            print(f"Total time: {end - start:.2f}s")
            print(f"Rewards: {_rewards}")
            print(f"Steps done: {int(steps_done)}")

            parameters = unmap_data(_params)
            rewards = unmap_data(_rewards)

            self._post_process_training(
                parameters,
                rewards,
                steps_done,
            )
            self.logger.write_logs(agents=self.agents, round_index=round_index)

            self._rank_agents()
            self._perform_exploit()
            self._perform_explore()
