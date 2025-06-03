# Copyright 2024 The Brax Authors.
#
import functools

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Metrics, Params, PRNGKey


State = envs.State
InferenceParams = tuple[running_statistics.NestedMeanStd, Params]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def init_hyperparameters(
    hyperparameters: dict,
    default_hyperparameters: dict,
    num_agents: int,
):
    for key, value in default_hyperparameters.items():
        if key not in hyperparameters:
            hyperparameters[key] = np.full(num_agents, value)

    return hyperparameters


def train_round(
    hyperparameters: dict,
    agent_params: dict,
    rng: PRNGKey,
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    unroll_length: int,
    num_envs: int,
    num_minibatches: int,
    batch_size: int,
    num_updates_per_batch,
    num_eval_envs: int,
    num_evals: int,
    deterministic_eval: bool,
    normalize_advantage: bool,
):
    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    num_training_steps_per_epoch = np.ceil(
        num_timesteps / env_step_per_training_step,
    ).astype(int)

    env = envs.training.wrap(environment, episode_length=episode_length)
    eval_env = envs.training.wrap(environment, episode_length=episode_length)
    eval_env = envs.training.EvalWrapper(eval_env)

    # Generate initial env states
    rng, key_env = jax.random.split(rng)
    key_envs = jax.random.split(key_env, num_envs)
    env_state = env.reset(key_envs)

    # Create the network
    ppo_network = ppo_networks.make_ppo_networks(
        env_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Create the optimizer
    optimizer = optax.adam(learning_rate=hyperparameters["learning_rate"])

    # Create the loss function
    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=hyperparameters["entropy_cost"],
        discounting=hyperparameters["discounting"],
        gae_lambda=hyperparameters["gae_lambda"],
        clipping_epsilon=hyperparameters["clipping_epsilon"],
        reward_scaling=hyperparameters["reward_scaling"],
        normalize_advantage=normalize_advantage,
    )

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn,
        optimizer,
        pmap_axis_name=None,
        has_aux=True,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry

        key, key_loss = jax.random.split(key)

        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state,
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry

        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )

        return (optimizer_state, params, key), metrics

    def training_rollout(
        carry: tuple[TrainingState, envs.State, PRNGKey],
        unused_t,
    ) -> tuple[tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry

        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (training_state.normalizer_params, training_state.params.policy),
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            data,
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )

        return (new_training_state, state, new_key), metrics

    def training_round(
        training_state: TrainingState,
        state: envs.State,
        key: PRNGKey,
    ) -> tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), metrics = jax.lax.scan(
            training_rollout,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        return training_state, state, metrics

    policy_params = agent_params.get("policy_params")
    if policy_params is None:
        rng, key_policy = jax.random.split(rng)
        policy_params = ppo_network.policy_network.init(key_policy)

    value_params = agent_params.get("value_params")
    if value_params is None:
        rng, key_value = jax.random.split(rng)
        value_params = ppo_network.value_network.init(key_value)

    normalizer_params = agent_params.get("normalizer_params")
    if normalizer_params is None:
        normalizer_params = running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32")),
        )

    init_params = ppo_losses.PPONetworkParams(policy=policy_params, value=value_params)

    optimizer_state = agent_params.get("optimizer_state")
    if optimizer_state is None:
        optimizer_state = optimizer.init(init_params)

    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=init_params,
        normalizer_params=normalizer_params,
        env_steps=0,
    )

    rng, eval_key = jax.random.split(rng)

    training_metrics = {}
    rng, rollout_key = jax.random.split(rng)

    training_state, env_state, training_metrics = training_round(
        training_state,
        env_state,
        rollout_key,
    )
    training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)

    def eval_rollout(carry, unused_t):
        """Generate one rollout in eval mode"""
        key = carry
        key, new_key = jax.random.split(key)
        reset_keys = jax.random.split(key, num_eval_envs)

        eval_first_state = eval_env.reset(reset_keys)

        eval_policy = make_policy(
            (training_state.normalizer_params, training_state.params.policy),
            deterministic=True,
        )

        final_state = acting.generate_unroll(
            eval_env,
            eval_first_state,
            eval_policy,
            key,
            unroll_length=episode_length,
        )[0]

        reward = final_state.info["eval_metrics"].episode_metrics["reward"]

        return new_key, reward

    _, rewards = jax.lax.scan(eval_rollout, eval_key, (), length=num_evals)
    eval_reward = jnp.mean(rewards)

    training_state_dict = {
        "policy_params": training_state.params.policy,
        "value_params": training_state.params.value,
        "normalizer_params": training_state.normalizer_params,
        "optimizer_state": training_state.optimizer_state,
    }

    steps_done = training_state.env_steps

    return training_state_dict, eval_reward, rng, steps_done
