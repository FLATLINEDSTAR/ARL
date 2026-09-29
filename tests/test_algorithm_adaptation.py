"""Native PPO/SAC adaptation uses only recorded post-shift data."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.logger import Logger

from adaptive_rl.algorithms.adaptation import (
    PPOAdaptationAdapter,
    SACAdaptationAdapter,
    run_adaptation_update,
)
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.benchmarking.adaptation_runtime import evaluate_episode
from adaptive_rl.protocol.adaptation import (
    PostShiftEpisode,
    Transition,
    UpdateBatch,
    build_update_batch,
)
from adaptive_rl.protocol.fork import model_fingerprint
from adaptive_rl.protocol.seeds import derive_seed


class _OneStepEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seed = seed
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(np.asarray(action, dtype=np.float32))
        return (
            np.zeros(3, dtype=np.float32),
            1.0,
            False,
            True,
            {"success": True, "collision": False},
        )


def _batch(algorithm, training_seed: int = 31001):
    model = algorithm.model
    assert model is not None
    episodes = []
    for episode_index in range(1, 6):
        transitions = []
        for step in range(3):
            observation = np.full(
                model.observation_space.shape,
                episode_index * 0.01 + step * 0.1,
                dtype=np.float32,
            )
            next_observation = observation + 0.05
            if isinstance(algorithm, PPOAlgorithm):
                obs_tensor, _ = model.policy.obs_to_tensor(observation)
                with torch.no_grad():
                    action_tensor, value_tensor, log_prob_tensor = model.policy(obs_tensor)
                    next_obs_tensor, _ = model.policy.obs_to_tensor(next_observation)
                    next_value_tensor = model.policy.predict_values(next_obs_tensor)
                action = action_tensor.cpu().numpy().reshape(-1)
                behavior_value = float(value_tensor.reshape(-1)[0].cpu())
                behavior_log_prob = float(log_prob_tensor.reshape(-1)[0].cpu())
                behavior_next_value = float(next_value_tensor.reshape(-1)[0].cpu())
            else:
                action = np.zeros(model.action_space.shape, dtype=np.float32)
                behavior_value = None
                behavior_log_prob = None
                behavior_next_value = None
            transitions.append(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(episode_index + step),
                    next_observation=next_observation,
                    terminated=False,
                    truncated=step == 2,
                    behavior_log_prob=behavior_log_prob,
                    behavior_value=behavior_value,
                    behavior_next_value=behavior_next_value,
                )
            )
        episodes.append(
            PostShiftEpisode(
                index=episode_index,
                seed=derive_seed(training_seed, "post", episode_index),
                transitions=tuple(transitions),
            )
        )
    return build_update_batch(training_seed, episodes, block_episode=5)


def _static_ppo_batch(seed: int) -> UpdateBatch:
    transitions = tuple(
        Transition(
            observation=np.full(3, index / 10.0, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=float(index % 4),
            next_observation=np.full(3, (index + 1) / 10.0, dtype=np.float32),
            terminated=index % 4 == 3,
            truncated=False,
            behavior_log_prob=-0.5,
            behavior_value=0.0,
        )
        for index in range(8)
    )
    return UpdateBatch(
        block_episode=5,
        seed=seed,
        visible_episode_indices=(1, 2, 3, 4, 5),
        transitions=transitions,
    )


def test_ppo_native_update_uses_recorded_rollout_without_environment_steps() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        batch = _batch(algorithm)
        model = algorithm.model
        assert model is not None
        model.set_logger(Logger(folder=None, output_formats=[]))
        model.logger.record("train/stale_loss", 999.0)
        before = model_fingerprint(algorithm)
        log = run_adaptation_update(algorithm, PPOAdaptationAdapter(), batch)
        assert log.loss_metrics
        assert "train/stale_loss" not in log.loss_metrics
        assert log.block_episode == 5
        assert log.transition_count == 15
        assert log.visible_episode_indices == (1, 2, 3, 4, 5)
        assert log.update_seed == batch.seed
        assert log.parameter_delta_l2 > 0.0
        assert log.fingerprint_before == before
        assert log.fingerprint_after != before
        assert model.num_timesteps == 0
    finally:
        env.close()


def test_ppo_buffer_gae_matches_hand_calculation_for_terminal_and_truncation() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        model = algorithm.model
        assert model is not None
        values = (0.1, 0.2, 0.3, 0.4)
        transitions = (
            Transition(
                observation=np.zeros(3, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=1.0,
                next_observation=np.ones(3, dtype=np.float32),
                terminated=False,
                truncated=False,
                behavior_log_prob=0.0,
                behavior_value=values[0],
            ),
            Transition(
                observation=np.ones(3, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=2.0,
                next_observation=np.full(3, 2.0, dtype=np.float32),
                terminated=False,
                truncated=True,
                behavior_log_prob=0.0,
                behavior_value=values[1],
                behavior_next_value=0.5,
            ),
            Transition(
                observation=np.full(3, 3.0, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=3.0,
                next_observation=np.full(3, 4.0, dtype=np.float32),
                terminated=False,
                truncated=False,
                behavior_log_prob=0.0,
                behavior_value=values[2],
            ),
            Transition(
                observation=np.full(3, 4.0, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=4.0,
                next_observation=np.full(3, 5.0, dtype=np.float32),
                terminated=True,
                truncated=False,
                behavior_log_prob=0.0,
                behavior_value=values[3],
            ),
        )
        batch = UpdateBatch(
            block_episode=5,
            seed=derive_seed(31001, "update", 0),
            visible_episode_indices=(1, 2, 3, 4, 5),
            transitions=transitions,
        )
        gamma = model.gamma
        gae_lambda = model.gae_lambda
        expected_last_truncated = 2.0 + gamma * 0.5 - values[1]
        expected_first_truncated = (
            1.0 + gamma * values[1] - values[0] + gamma * gae_lambda * expected_last_truncated
        )
        expected_last_terminated = 4.0 - values[3]
        expected_first_terminated = (
            3.0 + gamma * values[3] - values[2] + gamma * gae_lambda * expected_last_terminated
        )
        expected_advantages = np.asarray(
            [
                expected_first_truncated,
                expected_last_truncated,
                expected_first_terminated,
                expected_last_terminated,
            ],
            dtype=np.float32,
        )
        captured: dict[str, np.ndarray] = {}
        original_train = model.train

        def capture_buffer() -> None:
            captured["advantages"] = model.rollout_buffer.advantages.copy().reshape(-1)
            captured["returns"] = model.rollout_buffer.returns.copy().reshape(-1)
            model.logger.record("train/loss", 1.0)

        model.train = capture_buffer
        try:
            PPOAdaptationAdapter().update(algorithm, batch)
        finally:
            model.train = original_train
        np.testing.assert_allclose(captured["advantages"], expected_advantages, rtol=1e-6)
        np.testing.assert_allclose(
            captured["returns"], expected_advantages + np.asarray(values), rtol=1e-6
        )
    finally:
        env.close()


def test_ppo_update_restores_caller_rng_and_repeats_from_same_update_seed() -> None:
    envs = [gym.make("Pendulum-v1") for _ in range(3)]
    try:
        algorithms = [
            PPOAlgorithm(
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=2,
                seed=31001,
                device="cpu",
            )
            for env in envs
        ]
        assert len({model_fingerprint(algorithm) for algorithm in algorithms}) == 1
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        batch = _static_ppo_batch(derive_seed(31001, "update", 0))
        fingerprints = []
        for algorithm in algorithms[:2]:
            run_adaptation_update(algorithm, PPOAdaptationAdapter(), batch)
            fingerprints.append(model_fingerprint(algorithm))
        assert random.getstate() == python_state
        after_numpy = np.random.get_state()
        assert after_numpy[0] == numpy_state[0]
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        assert after_numpy[2:] == numpy_state[2:]
        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert fingerprints[0] == fingerprints[1]

        other_batch = _static_ppo_batch(derive_seed(31001, "update", 1))
        run_adaptation_update(algorithms[2], PPOAdaptationAdapter(), other_batch)
        assert model_fingerprint(algorithms[2]) != fingerprints[0]
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize(
    ("failure", "error_match"),
    [
        ("exception", "forced update failure"),
        ("nonfinite_parameter", "non-finite model state"),
        ("nonfinite_optimizer", "non-finite optimizer"),
        ("invalid_shape", "shape or dtype"),
    ],
)
def test_failed_ppo_update_restores_model_and_optimizer_state(
    failure: str, error_match: str
) -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        model = algorithm.model
        assert model is not None
        before_fingerprint = model_fingerprint(algorithm)
        before_num_timesteps = model.num_timesteps
        before_optimizer = model.policy.optimizer.state_dict()

        class FailingAdapter:
            def update(self, target, batch) -> None:
                del batch
                target_model = target.model
                parameter = next(target_model.policy.parameters())
                target_model.num_timesteps = 999
                if failure == "exception":
                    raise RuntimeError("forced update failure")
                if failure == "nonfinite_parameter":
                    parameter.data.fill_(float("nan"))
                elif failure == "nonfinite_optimizer":
                    target_model.policy.optimizer.state[parameter]["exp_avg"] = torch.full_like(
                        parameter, float("inf")
                    )
                else:
                    parameter.data = parameter.data.reshape(-1)[:-1]

        with pytest.raises((RuntimeError, FloatingPointError), match=error_match):
            run_adaptation_update(algorithm, FailingAdapter(), _static_ppo_batch(1))
        restored = algorithm.model
        assert restored is not None
        assert model_fingerprint(algorithm) == before_fingerprint
        assert restored.num_timesteps == before_num_timesteps
        assert restored.policy.optimizer.state_dict() == before_optimizer
    finally:
        env.close()


def test_sac_native_update_uses_fresh_post_only_replay_buffer() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(
            env=env,
            batch_size=4,
            learning_starts=0,
            gradient_steps=1,
            buffer_size=100,
            seed=31001,
            device="cpu",
        )
        batch = _batch(algorithm)
        model = algorithm.model
        assert model is not None
        original_buffer = model.replay_buffer
        before = model_fingerprint(algorithm)
        native_train = model.train
        replay_snapshot: dict[str, np.ndarray] = {}

        def inspect_replay(gradient_steps: int, batch_size: int) -> None:
            replay = model.replay_buffer
            assert replay is not None
            size = replay.size()
            replay_snapshot["observations"] = replay.observations[:size, 0].copy()
            replay_snapshot["rewards"] = replay.rewards[:size, 0].copy()
            replay_snapshot["dones"] = replay.dones[:size, 0].copy()
            replay_snapshot["timeouts"] = replay.timeouts[:size, 0].copy()
            native_train(gradient_steps=gradient_steps, batch_size=batch_size)

        model.train = inspect_replay
        try:
            log = run_adaptation_update(algorithm, SACAdaptationAdapter(), batch)
        finally:
            model.train = native_train
        assert log.loss_metrics
        assert log.block_episode == 5
        assert log.transition_count == 15
        assert log.visible_episode_indices == (1, 2, 3, 4, 5)
        assert log.update_seed == batch.seed
        assert log.parameter_delta_l2 > 0.0
        assert log.fingerprint_before == before
        assert log.fingerprint_after != before
        assert model.replay_buffer is original_buffer
        assert original_buffer is not None and original_buffer.size() == 0
        assert model.num_timesteps == 0
        np.testing.assert_allclose(
            replay_snapshot["observations"],
            np.stack([transition.observation for transition in batch.transitions]),
        )
        np.testing.assert_allclose(
            replay_snapshot["rewards"], [transition.reward for transition in batch.transitions]
        )
        np.testing.assert_array_equal(
            replay_snapshot["dones"],
            [
                float(transition.terminated or transition.truncated)
                for transition in batch.transitions
            ],
        )
        np.testing.assert_array_equal(
            replay_snapshot["timeouts"],
            [
                float(transition.truncated and not transition.terminated)
                for transition in batch.transitions
            ],
        )
    finally:
        env.close()


def test_episode_collector_records_transition_and_keeps_policy_constant() -> None:
    model_env = gym.make("Pendulum-v1")
    try:
        algorithm = PPOAlgorithm(
            env=model_env, n_steps=8, batch_size=4, n_epochs=1, seed=31001, device="cpu"
        )
        env = _OneStepEnv()
        before = model_fingerprint(algorithm)
        record = evaluate_episode(
            algorithm=algorithm,
            env=env,
            training_seed=31001,
            phase="post",
            episode_index=1,
            episode_seed=derive_seed(31001, "post", 1),
            algorithm_name="ppo",
            environment_name="fake_drone",
            deterministic=True,
            arm="shared",
        )
        assert env.reset_seed == record.episode_seed
        assert record.length == 1
        assert record.success is True
        assert record.collision is False
        assert len(record.transitions) == 1
        assert record.transitions[0].behavior_log_prob is not None
        assert record.transitions[0].behavior_value is not None
        assert record.transitions[0].behavior_next_value is not None
        assert record.policy_fingerprint_start == record.policy_fingerprint_end == before
        assert model_fingerprint(algorithm) == before
    finally:
        model_env.close()
