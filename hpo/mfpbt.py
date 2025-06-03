import functools
import os
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from brax import envs

from hpo.utils import explore_functions
from hpo.utils.logger import Logger
from hpo.utils.mf_pbt_genetics import genetics
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
    population_index: int
    parent_hps: int
    parent_network: int
    hyperparameters: dict
    params: dict
    reward: float
    env_steps: int


@dataclass
class Population:
    index: int
    frequency: int
    agents: list[Agent]


class MultipleFrequenciesPopulationBasedTraining:
    def __init__(
        self,
        num_agents: int,
        frequencies: list[int],
        exp_name: str,
        env_name: str,
        num_envs_per_agent: int,
        num_timesteps_round: int,
        num_rounds: int,
        hpo_search_space: dict,
        default_hps: dict,
        training_config: dict,
        routines: dict,
        logging_directory: str = None,
        backend: str = "positional",
        numpy_seed: int = 0,
        jax_seed: int = 0,
    ) -> None:
        self.numpy_rng = np.random.default_rng(numpy_seed)
        self.jax_seed = jax_seed
        self.num_agents = num_agents
        self.frequencies = frequencies
        self.exp_name = exp_name
        self.num_rounds = num_rounds
        self.hpo_search_space = hpo_search_space
        self.default_hps = default_hps
        self.training_config = training_config

        # Population management
        self.num_populations = len(frequencies)
        self.num_agents_per_population = num_agents // self.num_populations

        self.agents = None
        self.populations = None

        assert num_agents % self.num_populations == 0, "For now, we expect all the populations to be of the same size"

        assert self.num_agents_per_population % 4 == 0, (
            "In order to apply the MF-PBT algorithm we need to split populations in 4."
        )

        # Load env
        env = envs.get_environment(env_name=env_name, backend=backend)
        self.training_config.update(
            {
                "environment": env,
                "num_timesteps": num_timesteps_round,
                "num_envs": num_envs_per_agent,
            },
        )

        self._assert_config()

        # Routines management
        assert routines["explore"] is None or hpo_search_space.keys() == routines["explore"].keys(), (
            "An exploration function AND a search space mustbe specified for all hyperparameters you wish to optimize"
        )

        # Do nothing cases
        if hpo_search_space is None:
            hpo_search_space = {}
        self.hpo_keys = list(hpo_search_space.keys())
        self.param_keys = [
            "policy_params",
            "value_params",
            "normalizer_params",
            "optimizer_state",
        ]

        self._explore = {
            key: getattr(explore_functions, routines["explore"][key]["function_name"])(
                **routines["explore"][key]["function_params"],
            )
            for key in self.hpo_keys
        }

        # Jax device management
        self.num_devices = jax.device_count()
        devices = jax.devices()
        print(f"Local devices: {self.num_devices}")
        print(f"Devices: {devices}")

        self.num_agents_per_device = num_agents // self.num_devices

        self._map_data = functools.partial(
            map_data,
            num_devices=self.num_devices,
            num_agents_per_device=self.num_agents_per_device,
        )

        # Logging management
        if logging_directory is not None:
            self.logging_path = os.path.join(logging_directory, exp_name)
        else:
            self.logging_path = os.path.join(PATH_TO_MAIN_PROJECT, "runs", exp_name)

        self.logger = Logger(
            logging_path=self.logging_path,
            hpo_keys=self.hpo_keys,
        )

        self.global_ranking_indexes = None
        self.per_population_ranking_indexes = None
        self.starting_round = None
        self._initiate_experiment()

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

    def _flatten_populations(self):
        return list(chain(*[population.agents for population in self.populations]))

    def _unflatten_populations(self, agents: list[Agent]):
        for population_index in range(self.num_populations):
            self.populations[population_index].agents = agents[
                population_index * self.num_agents_per_population : (population_index + 1)
                * self.num_agents_per_population
            ]

    def _initiate_experiment(self):
        # This method let us know if we are reloading a previous xp to continue it
        reloading, round_index, log_df, agent_params = self.logger.read_logs()

        if not reloading:
            print(f"New experiment: {self.exp_name}\n")
            self.populations = [self._initiate_population(i) for i in range(self.num_populations)]
            self.starting_round = 0

        else:
            print(f"Reloading experiment {self.exp_name} from round {round_index}\n")
            self.starting_round = round_index + 1
            self.populations = [self._initiate_population(i) for i in range(self.num_populations)]
            self._reload_agents(log_df, agent_params, round_index)

    def _initiate_population(self, population_index: int):
        initial_hps = self._init_hps()
        initial_params = {key: None for key in self.param_keys}
        # Convention at start of experiment agents have no parent
        return Population(
            index=population_index,
            frequency=self.frequencies[population_index],
            agents=[
                Agent(
                    index=i,
                    population_index=population_index,
                    parent_hps=-1,
                    parent_network=-1,
                    hyperparameters=initial_hps[i],
                    params=initial_params,
                    reward=0.0,
                    env_steps=0,
                )
                for i in range(self.num_agents_per_population)
            ],
        )

    def _init_hps(self):
        # We initiate hps randomly between two values
        # defined in the hpo_search_space dictionary
        values = {key: self._get_init_values(key) for key in self.hpo_keys}
        initial_hps = [{key: values[key][i] for key in self.hpo_keys} for i in range(self.num_agents_per_population)]
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
                self.numpy_rng.uniform(
                    value_min,
                    value_max,
                    self.num_agents_per_population,
                ),
            )

        elif method == "uniform":
            return self.numpy_rng.uniform(
                value_min,
                value_max,
                self.num_agents_per_population,
            )

        elif method == "log_grid":
            return np.logspace(value_min, value_max, self.num_agents_per_population)
        else:
            raise NotImplementedError("I will code other init methods as needed")

    def _reload_agents(self, log_df, agent_params, round_index):
        for i in range(len(log_df)):
            line = log_df.iloc[i]
            index = int(line["index"])
            population_index = int(line["population_index"])

            agent = self.populations[population_index].agents[index]
            agent.env_steps = int(line["env_steps"])
            agent.reward = line["reward"]
            for hp in self.hpo_keys:
                agent.hyperparameters[hp] = line[hp]

            agent.params = agent_params[index + population_index * self.num_agents_per_population]

        # Then we reperform the evolution step
        self._rank_agents()
        self._perform_genetics(round_index)

    def _format_hyperparameters(self):
        # Get all hyperparameters in the good format
        optimizing_hyperparameters = {}
        agents = self._flatten_populations()
        for hp in self.hpo_keys:
            values = np.array([agent.hyperparameters[hp] for agent in agents])
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

        agents = self._flatten_populations()
        return jax.tree_map(
            lambda *arrays: jnp.stack(arrays),
            *[agent.params for agent in agents],
        )

    def _post_process_training(self, parameters, rewards, steps_done):
        agents = self._flatten_populations()
        for global_index in range(self.num_agents):
            agents[global_index].reward = rewards[global_index]
            agents[global_index].params = jax.tree_map(
                lambda x: x[global_index],
                parameters,
            )
            agents[global_index].env_steps += int(steps_done)

        self._unflatten_populations(agents)

    def _rank_agents(self):
        global_rewards = np.array(
            [agent.reward for agent in self._flatten_populations()],
        )
        self.global_ranking_indexes = np.argsort(-global_rewards)

        local_rewards = np.array(
            [[agent.reward for agent in population.agents] for population in self.populations],
        )

        self.local_ranking_indexes = [
            np.argsort(-local_rewards[population_index]) for population_index in range(self.num_populations)
        ]

    def _perform_genetics(self, round_index: int):
        agents = self._flatten_populations()
        parents_hps, parents_network, need_explore = genetics(
            self.global_ranking_indexes,
            self.local_ranking_indexes,
            round_index,
            self.frequencies,
        )

        for global_index, agent in enumerate(agents):
            agent.parent_hps = parents_hps[global_index]
            agent.parent_network = parents_network[global_index]

            # Update network
            if global_index != parents_network[global_index]:
                agent.params = deepcopy(agents[parents_network[global_index]].params)

            # Update hps
            if global_index != parents_hps[global_index]:
                parent = agents[parents_hps[global_index]]

                for key in self.hpo_keys:
                    value = parent.hyperparameters[key]
                    new_value = self._explore[key](value) if need_explore[global_index] else value
                    agent.hyperparameters[key] = new_value
        self._unflatten_populations(agents)

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
            try:
                out = pmapped_train_fn(hyperparameters, parameters, key_rounds)
            except jaxlib.xla_extension.XlaRuntimeError as e:
                print("The execution of one round is taking too much memory for the current VRAM of the GPU.\n" \
                "Please consider reducing some of the following parameters: num_agents, num_envs_per_agent, batch_size, num_eval_envs.")
                break

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

            self._post_process_training(parameters, rewards, steps_done)
            agents = self._flatten_populations()
            self.logger.write_logs(agents=agents, round_index=round_index)

            self._rank_agents()
            self._perform_genetics(round_index)
