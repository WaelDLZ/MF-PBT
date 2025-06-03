import functools
from typing import Tuple, List

import numpy as np

# Constants
POPULATION_QUARTERS = 4  # Number of segments to split population into
BOTTOM_QUARTER_RATIO = 0.25  # Fraction of worst agents to replace
MIGRATION_BRACKET_START = 2  # Which quarter starts migration (0-indexed)
MIGRATION_BRACKET_END = 3  # Which quarter ends migration (0-indexed)


def genetics(
    global_ranking_indexes: np.ndarray,
    local_ranking_indexes: List[np.ndarray],
    round_index: int,
    frequencies: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform MF-PBT genetic algorithm operations.

    Args:
        global_ranking_indexes: Global agent ranking (best to worst)
        local_ranking_indexes: Per-population agent rankings
        round_index: Current training round
        frequencies: Evolution frequencies for each population

    Returns:
        Tuple of (parents_hps, parents_network, need_explore)
    """
    # Determine which populations should evolve this round
    num_populations = len(frequencies)
    num_agents_per_population = len(local_ranking_indexes[0])
    num_agents = len(global_ranking_indexes)

    concerned_populations = _get_evolving_populations(round_index, frequencies)
    print(f"Concerned populations: {concerned_populations}\n")

    # Initialize inheritance arrays
    parents_hps, parents_network, need_explore = _initialize_inheritance_arrays(
        num_agents
    )

    # Create ranking utilities
    inverse_ranking = _create_inverse_ranking(global_ranking_indexes, num_agents)

    # Helper functions for index conversion
    index_converter = IndexConverter(num_agents_per_population)

    # Perform internal exploitation for concerned populations
    for population_index in range(num_populations):
        if concerned_populations[population_index]:
            _perform_internal_exploit(
                population_index,
                local_ranking_indexes[population_index],
                parents_hps,
                parents_network,
                need_explore,
                index_converter,
            )

    # Perform migration between populations
    for population_index in range(num_populations):
        if concerned_populations[population_index]:
            _perform_migration(
                population_index,
                local_ranking_indexes[population_index],
                global_ranking_indexes,
                parents_hps,
                parents_network,
                inverse_ranking,
                frequencies,
                index_converter,
            )

    return parents_hps, parents_network, need_explore


def _get_evolving_populations(round_index: int, frequencies: List[int]) -> List[bool]:
    """Determine which populations should evolve this round."""
    return [(round_index + 1) % freq == 0 for freq in frequencies]


def _initialize_inheritance_arrays(
    num_agents: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize arrays for tracking inheritance and exploration needs."""
    parents_hps = np.arange(num_agents)
    parents_network = np.arange(num_agents)
    need_explore = np.zeros(num_agents, dtype=bool)
    return parents_hps, parents_network, need_explore


def _create_inverse_ranking(
    global_ranking_indexes: np.ndarray, num_agents: int
) -> np.ndarray:
    """Create inverse ranking for reward comparison."""
    inverse_ranking = np.empty_like(global_ranking_indexes)
    for i in range(num_agents):
        inverse_ranking[global_ranking_indexes[i]] = num_agents - i
    return inverse_ranking


class IndexConverter:
    """Helper class for converting between local and global agent indices."""

    def __init__(self, num_agents_per_population: int):
        self.num_agents_per_population = num_agents_per_population

    def local_to_global(self, agent_index: int, population_index: int) -> int:
        """Convert local agent index to global index."""
        return agent_index + self.num_agents_per_population * population_index

    def global_to_local(self, global_index: int) -> Tuple[int, int]:
        """Convert global index to (population_index, agent_index)."""
        return divmod(global_index, self.num_agents_per_population)

    def get_population(self, global_index: int) -> int:
        """Get population index from global agent index."""
        return self.global_to_local(global_index)[0]


def _perform_internal_exploit(
    population_index: int,
    local_ranking: np.ndarray[int],
    parents_hps: np.ndarray,
    parents_network: np.ndarray,
    need_explore: np.ndarray[bool],
    index_converter: IndexConverter,
):
    """Perform internal exploitation within a population."""
    num_agents_per_population = len(local_ranking)
    share = num_agents_per_population // POPULATION_QUARTERS
    parents = np.arange(num_agents_per_population)
    parents[local_ranking[-share:]] = local_ranking[:share]

    glob = functools.partial(
        index_converter.local_to_global,
        population_index=population_index,
    )
    for agent in range(num_agents_per_population):
        parent = parents[agent]
        if parent != agent:
            need_explore[glob(agent)] = True
            parents_hps[glob(agent)] = glob(parent)
            parents_network[glob(agent)] = glob(parent)


def _perform_migration(
    population_index: int,
    local_ranking: np.ndarray[int],
    global_ranking: np.ndarray[int],
    parents_hps: np.ndarray,
    parents_network: np.ndarray,
    inverse_ranking: np.ndarray,
    frequencies: List[int],
    index_converter: IndexConverter,
):
    """Perform migration between populations."""
    num_agents_per_population = len(local_ranking)
    share = num_agents_per_population // POPULATION_QUARTERS
    migration_bracket = local_ranking[
        MIGRATION_BRACKET_START * share : MIGRATION_BRACKET_END * share
    ]

    migration_bracket = [
        index_converter.local_to_global(agent, population_index)
        for agent in migration_bracket
    ]
    external_ranking = [
        agent
        for agent in global_ranking
        if index_converter.get_population(agent) != population_index
    ]

    def perform_transfer(agent: int, migrant: int):
        """Perform the transfer of networks and hyperparameters between agents."""
        parents_network[agent] = migrant

        in_population = index_converter.get_population(agent)
        out_population = index_converter.get_population(migrant)

        if frequencies[in_population] < frequencies[out_population]:
            parents_hps[agent] = migrant
        elif frequencies[in_population] > frequencies[out_population]:
            best_internal_agent = local_ranking[0]
            parents_hps[agent] = index_converter.local_to_global(
                best_internal_agent,
                in_population,
            )

    for agent in migration_bracket:
        migrant = external_ranking[0]
        if inverse_ranking[agent] < inverse_ranking[migrant]:
            perform_transfer(agent, migrant)
            external_ranking.pop(0)
