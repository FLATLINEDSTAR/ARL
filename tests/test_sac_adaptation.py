"""Isolation, ordering, and audit checks for the SAC adaptation treatment."""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from adaptive_rl.algorithms.adaptation import (
    SACAdaptationAdapter,
    TaggedReplayBuffer,
    run_adaptation_update,
)
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.benchmarking.adaptation_artifacts import CSV_FIELDS
from adaptive_rl.benchmarking.adaptation_runner import (
    _build_tagged_sac_batch,
    _reset_sac_adaptation_buffer,
    run_adaptation_benchmark,
)
from adaptive_rl.benchmarking.adaptation_runtime import EpisodeTaggedTransition
from adaptive_rl.config import load_config
from adaptive_rl.protocol.adaptation import PostShiftEpisode, Transition, UpdateBatch
from adaptive_rl.protocol.fork import FrozenPolicy, clone_algorithm, model_fingerprint
from adaptive_rl.protocol.seeds import derive_seed


def _episodes() -> list[PostShiftEpisode]:
    episodes = []
    for index in range(1, 6):
        transitions = tuple(
            Transition(
                observation=np.full(3, index * 0.01 + step * 0.1, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=float(index + step),
                next_observation=np.full(3, index * 0.01 + (step + 1) * 0.1, dtype=np.float32),
                terminated=False,
                truncated=step == 1,
            )
            for step in range(2)
        )
        episodes.append(
            PostShiftEpisode(
                index=index,
                seed=derive_seed(31001, "post", index),
                transitions=transitions,
            )
        )
    return episodes


def _tag_transition(transition: Transition, episode_index: int) -> EpisodeTaggedTransition:
    return EpisodeTaggedTransition(
        observation=transition.observation,
        action=transition.action,
        reward=transition.reward,
        next_observation=transition.next_observation,
        terminated=transition.terminated,
        truncated=transition.truncated,
        environment_action=transition.environment_action,
        behavior_log_prob=transition.behavior_log_prob,
        behavior_value=transition.behavior_value,
        behavior_next_value=transition.behavior_next_value,
        episode_index=episode_index,
    )


def test_sac_runner_batch_has_only_tagged_visible_post_shift_data() -> None:
    batch = _build_tagged_sac_batch(31001, _episodes(), block_episode=5)
    assert batch.visible_episode_indices == (1, 2, 3, 4, 5)
    assert [item.episode_index for item in batch.transitions] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert batch.seed == derive_seed(31001, "update", 0)

    with pytest.raises(RuntimeError, match="requires completed episodes 1..5"):
        _build_tagged_sac_batch(31001, _episodes() + _episodes()[:1], block_episode=5)


def test_sac_adapter_rejects_future_or_missing_episode_tags() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(env=env, batch_size=4, seed=31001, device="cpu")
        good = _build_tagged_sac_batch(31001, _episodes(), block_episode=5)
        future = _tag_transition(good.transitions[-1], 6)
        future_batch = UpdateBatch(
            block_episode=5,
            seed=good.seed,
            visible_episode_indices=good.visible_episode_indices,
            transitions=(*good.transitions[:-1], future),
        )
        with pytest.raises(RuntimeError, match="invisible episode"):
            SACAdaptationAdapter().update(algorithm, future_batch)

        missing_episode = UpdateBatch(
            block_episode=5,
            seed=good.seed,
            visible_episode_indices=good.visible_episode_indices,
            transitions=tuple(item for item in good.transitions if item.episode_index != 3),
        )
        with pytest.raises(RuntimeError, match="every visible episode"):
            SACAdaptationAdapter().update(algorithm, missing_episode)
    finally:
        env.close()


def test_adaptation_start_purges_a_copied_nominal_replay_buffer() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(env=env, batch_size=4, seed=31001, device="cpu")
        model = algorithm.model
        assert model is not None and model.replay_buffer is not None
        original = model.replay_buffer
        shape = model.observation_space.shape
        original.add(
            np.full((1, *shape), 77.0, dtype=np.float32),
            np.full((1, *shape), 78.0, dtype=np.float32),
            np.zeros((1, *model.action_space.shape), dtype=np.float32),
            np.asarray([9876.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            infos=[{"TimeLimit.truncated": False}],
        )
        adaptive = clone_algorithm(algorithm)
        _reset_sac_adaptation_buffer(adaptive)
        assert adaptive.model.replay_buffer.size() == 0
        assert model.replay_buffer.size() == 1
    finally:
        env.close()


def test_sac_block_uses_isolated_tagged_buffer_and_emits_complete_audit() -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(
            env=env,
            batch_size=4,
            learning_starts=0,
            gradient_steps=1,
            buffer_size=64,
            seed=31001,
            device="cpu",
        )
        model = algorithm.model
        assert model is not None and model.replay_buffer is not None
        nominal = model.replay_buffer
        shape = model.observation_space.shape
        nominal.add(
            np.full((1, *shape), 999.0, dtype=np.float32),
            np.full((1, *shape), 998.0, dtype=np.float32),
            np.zeros((1, *model.action_space.shape), dtype=np.float32),
            np.asarray([123456.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            infos=[{"TimeLimit.truncated": False}],
        )
        batch = _build_tagged_sac_batch(31001, _episodes(), block_episode=5)
        original_train = model.train
        captured: dict[str, object] = {}

        def inspect_train(*, gradient_steps: int, batch_size: int) -> None:
            replay = model.replay_buffer
            assert isinstance(replay, TaggedReplayBuffer)
            assert replay.size() == 10
            assert replay.episode_tags == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            size = replay.size()
            assert 123456.0 not in replay.rewards[:size, 0]
            captured["tags"] = list(replay.episode_tags)
            original_train(gradient_steps=gradient_steps, batch_size=batch_size)

        model.train = inspect_train
        before = model_fingerprint(algorithm)
        try:
            log = run_adaptation_update(algorithm, SACAdaptationAdapter(), batch)
        finally:
            model.train = original_train

        assert log.fingerprint_before == before
        assert log.fingerprint_after != before
        assert log.parameter_delta_l2 > 0.0
        assert model.replay_buffer is nominal
        assert nominal.size() == 1
        assert captured["tags"] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        assert model._adaptive_rl_last_train_call_count == 1
        audit = model._adaptive_rl_last_audit
        assert audit["gradient_steps"] == 1
        assert audit["buffer_size"] == 10
        assert audit["actor_loss"] is not None
        assert audit["critic_loss"] is not None
        assert audit["alpha"] > 0
        assert audit["target_network_update_count"] == 1
        assert audit["polyak_taus_observed"] == [model.tau]
        assert audit["fingerprint_before"] != audit["fingerprint_after"]
    finally:
        env.close()


def test_fixed_sac_facade_never_trains_or_changes_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = gym.make("Pendulum-v1")
    try:
        algorithm = SACAlgorithm(env=env, batch_size=4, seed=31001, device="cpu")
        model = algorithm.model
        assert model is not None
        fixed = FrozenPolicy(algorithm)
        before = fixed.fingerprint
        parameter_snapshots = {
            name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()
        }
        calls = 0

        def forbidden_train(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("Fixed SAC called train")

        monkeypatch.setattr(model, "train", forbidden_train)
        for optimizer in (
            model.actor.optimizer,
            model.critic.optimizer,
            model.ent_coef_optimizer,
        ):
            if optimizer is not None:
                monkeypatch.setattr(optimizer, "step", forbidden_train)
        for _ in range(15):
            fixed.predict(np.zeros(3, dtype=np.float32), deterministic=True)
            calls += 1
        assert calls == 15
        assert fixed.fingerprint == before
        assert fixed.parameter_delta_l2 == 0.0
        assert model_fingerprint(algorithm) == before
        assert all(
            torch.equal(parameter_snapshots[name], value.detach().cpu())
            for name, value in model.policy.state_dict().items()
        )
    finally:
        env.close()


def test_sac_smoke_runner_emits_ten_audited_blocks_and_ppo_csv_schema(
    tmp_path: Path,
) -> None:
    config = load_config("configs/drone_distribution_shift.yaml")
    algorithm = config.algorithm.model_copy(deep=True)
    algorithm.name = "sac"
    algorithm.parameters = {"learning_starts": 1, "gradient_steps": 1, "buffer_size": 64}
    config.algorithm = algorithm
    artifact = run_adaptation_benchmark(
        config,
        output_dir=tmp_path,
        training_seeds=[31001],
        smoke=True,
    )
    replicate = artifact["replicates"][0]
    assert replicate["status"] == "completed"
    blocks = replicate["update_blocks"]
    assert len(blocks) == 10
    assert [block["block_episode"] for block in blocks] == list(range(5, 15))
    audits = [block["sac_audit"] for block in blocks]
    assert all(audit["buffer_size"] > 0 for audit in audits)
    assert [audit["fingerprint_after"] for audit in audits[:-1]] == [
        audit["fingerprint_before"] for audit in audits[1:]
    ]
    with (tmp_path / "adaptation.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.reader(handle).__next__()) == CSV_FIELDS

    repeated = run_adaptation_benchmark(
        config,
        output_dir=tmp_path / "same-seed",
        training_seeds=[31001],
        smoke=True,
    )["replicates"][0]
    assert repeated["frozen_fingerprint"] == replicate["frozen_fingerprint"]
    for phase in (
        "shared_pre_shift_episodes",
        "shared_shock_episodes",
        "adaptive_episodes",
        "fixed_episodes",
    ):
        assert [item["reward"] for item in repeated[phase]] == [
            item["reward"] for item in replicate[phase]
        ]

    other_seed = run_adaptation_benchmark(
        config,
        output_dir=tmp_path / "different-seed",
        training_seeds=[31002],
        smoke=True,
    )["replicates"][0]
    assert other_seed["frozen_fingerprint"] != replicate["frozen_fingerprint"]
