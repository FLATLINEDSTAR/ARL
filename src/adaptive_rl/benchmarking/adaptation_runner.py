"""Protocol-ordered Issue #265 train/freeze/share/fork experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.adaptation import (
    PPOAdaptationAdapter,
    SACAdaptationAdapter,
    run_adaptation_update,
)
from adaptive_rl.benchmarking.adaptation_artifacts import (
    REPLICATE_CHECKPOINT_SCHEMA_VERSION,
    STUDY_ARTIFACT_SCHEMA_VERSION,
    STUDY_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    read_replicate_checkpoint,
    validate_study_manifest,
    write_adaptation_artifacts,
    write_adaptive_vs_fixed_artifacts,
    write_or_verify_study_manifest,
    write_replicate_checkpoint,
    write_study_manifest,
)
from adaptive_rl.benchmarking.adaptation_runtime import EpisodeRecord, evaluate_episode
from adaptive_rl.benchmarking.adaptation_statistics import analyze_primary_cells
from adaptive_rl.config import ExperimentConfig, compute_config_sha256
from adaptive_rl.environments.registry import make_env
from adaptive_rl.protocol.adaptation import (
    AdaptationAdapter,
    Transition,
    build_update_batch,
    validate_block_sequence,
)
from adaptive_rl.protocol.constants import (
    ISSUE271_CONFIG_SHA256,
    ISSUE271_V2_CONFIG_SHA256,
    ISSUE271_TREATMENT_CARD_SHA256,
    K_PRE,
    N_POST,
    PRIMARY_CELLS,
    TRAINING_SEEDS,
)
from adaptive_rl.protocol.fork import fork_adaptive_and_fixed, model_fingerprint
from adaptive_rl.protocol.recovery import compute_recovery
from adaptive_rl.protocol.seeds import frozen_schedule, schedule_fingerprint
from adaptive_rl.protocol.statistics import decide_family
from adaptive_rl.training.trainer import get_trainer

logger = logging.getLogger(__name__)

EnvironmentFactory = Callable[..., gym.Env]
TrainerFactory = Callable[..., Any]


def _repository_metadata() -> dict[str, Any]:
    commit: str | None
    dirty: bool | None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for distribution in ("adaptive-rl", "gymnasium", "stable-baselines3", "torch", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    runtime_settings: dict[str, Any] = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    }
    runtime_settings.update(
        {
            "torch_num_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    )
    return {
        "repository_commit": commit,
        "working_tree_dirty": dirty,
        "runtime_versions": versions,
        "runtime_settings": runtime_settings,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "research" / "TREATMENT_CARD.md"


def _new_env(
    config: ExperimentConfig,
    *,
    shifted: bool,
    environment_factory: EnvironmentFactory,
    max_steps: Optional[int] = None,
) -> gym.Env:
    parameters = dict(config.environment.parameters)
    parameters["max_steps"] = int(max_steps or config.environment.max_steps)
    if shifted:
        benchmark = config.adaptation_benchmark
        if benchmark is None:
            raise ValueError("configuration does not declare the Issue #265 adaptation cell")
        parameters.update(benchmark.shift_parameters)
    return environment_factory(config.environment.name, **parameters)


def _adapter_for(algorithm_name: str) -> AdaptationAdapter:
    if algorithm_name == "ppo":
        return PPOAdaptationAdapter()
    if algorithm_name == "sac":
        return SACAdaptationAdapter()
    raise ValueError(f"Issue #265 supports PPO and SAC, got {algorithm_name!r}")


def _train_once(
    config: ExperimentConfig,
    training_seed: int,
    training_dir: Path,
    *,
    trainer_factory: TrainerFactory,
    environment_factory: EnvironmentFactory,
    smoke: bool,
) -> tuple[Any, dict[str, Any]]:
    if config.training is None:
        raise ValueError("Issue #265 requires a training configuration")
    if training_dir.exists():
        raise FileExistsError(f"training output already exists: {training_dir}")
    training_dir.mkdir(parents=True)

    effective = config.model_copy(deep=True)
    training = effective.training
    if training is None:
        raise ValueError("Issue #265 requires a training configuration")
    effective.seed = int(training_seed)
    effective.name = f"{config.name}_seed_{training_seed}"
    effective.output_dir = training_dir
    effective.log_dir = training_dir / "logs"
    # The training component does not receive the TEST-B scenario payload.
    effective.adaptation_benchmark = None
    algorithm_config = effective.algorithm.model_copy(deep=True)
    algorithm_parameters = dict(algorithm_config.parameters)
    algorithm_parameters["seed"] = int(training_seed)
    if smoke:
        training.total_timesteps = min(training.total_timesteps, 32)
        if algorithm_config.name.lower() == "ppo":
            algorithm_parameters.update({"n_steps": 16, "n_epochs": 1})
            algorithm_config.batch_size = min(8, algorithm_config.batch_size)
        else:
            algorithm_parameters.update(
                {"learning_starts": 1, "gradient_steps": 1, "buffer_size": 64}
            )
            algorithm_config.batch_size = min(8, algorithm_config.batch_size)
    algorithm_config.parameters = algorithm_parameters
    effective.algorithm = algorithm_config
    training_env = _new_env(
        effective,
        shifted=False,
        environment_factory=environment_factory,
        max_steps=8 if smoke else None,
    )
    trainer = None
    started = time.perf_counter()
    try:
        trainer = trainer_factory(config=effective, env=training_env)
        result = trainer.fit()
        algorithm = trainer.algorithm
    except BaseException:
        if trainer is not None:
            trainer.close()
        else:
            training_env.close()
        raise
    else:
        training_seconds = float(time.perf_counter() - started)
        trainer.close()
    return algorithm, {
        "training_seed": training_seed,
        "total_timesteps_requested": training.total_timesteps,
        "total_timesteps_completed": int(getattr(algorithm, "num_timesteps", 0)),
        "training_time_seconds": training_seconds,
        "model_path": str(result.final_model_path),
        "training_metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "episodes_completed": int(result.episodes_completed),
        "mean_reward": float(result.mean_reward),
        "smoke_override": bool(smoke),
        "effective_config": effective.model_dump(mode="json"),
    }


def _run_evaluation_segment(
    *,
    algorithm: Any,
    env: gym.Env,
    training_seed: int,
    algorithm_name: str,
    environment_name: str,
    phase: str,
    indices: Sequence[int],
    seeds: Sequence[int],
    deterministic: bool,
    arm: str,
    initial_update_log: Any = None,
) -> list[EpisodeRecord]:
    if len(indices) != len(seeds):
        raise ValueError("episode indices and seeds must have matching lengths")
    records: list[EpisodeRecord] = []
    update_log = initial_update_log
    for episode_index, episode_seed in zip(indices, seeds):
        records.append(
            evaluate_episode(
                algorithm=algorithm,
                env=env,
                training_seed=training_seed,
                phase=phase,
                episode_index=episode_index,
                episode_seed=episode_seed,
                algorithm_name=algorithm_name,
                environment_name=environment_name,
                deterministic=deterministic,
                arm=arm,
                update_log=update_log,
            )
        )
        update_log = None
    return records


def _recover(pre: Sequence[EpisodeRecord], post: Sequence[EpisodeRecord]) -> dict[str, Any]:
    result = compute_recovery(
        [episode.reward for episode in pre], [episode.reward for episode in post]
    )
    output = asdict(result)
    output["T_H"] = result.truncated_recovery_time
    return output


def _return_vector_fingerprint(episodes: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps([float(item["reward"]) for item in episodes], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _enable_study_determinism() -> dict[str, Any]:
    """Enable deterministic Torch behavior where supported and report limits."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "limitations": [
            "warn-only Torch operations may remain nondeterministic",
            "cross-hardware and cross-library-version bitwise identity is not claimed",
            "training and update RNGs are reseeded from the recorded protocol seeds",
        ],
    }


def _audit_replicate_invariants(
    replicate: dict[str, Any], schedule: dict[int, dict[str, list[int]]]
) -> dict[str, Any]:
    """Recompute the execution invariants from one serialized replicate record."""
    blocks = replicate["update_blocks"]
    arms = replicate["arm_input_fingerprints"]
    pre_hash = _return_vector_fingerprint(replicate["shared_pre_shift_episodes"])
    shock_hash = _return_vector_fingerprint(replicate["shared_shock_episodes"])
    checks = {
        "pre_shift_shared_once": len(replicate["shared_pre_shift_episodes"]) == K_PRE,
        "pre_shift_identical_across_arms": (
            arms["adaptive"]["pre"] == pre_hash == arms["fixed"]["pre"]
        ),
        "shock_shared_once": len(replicate["shared_shock_episodes"]) == 5,
        "shock_returns_identical_across_arms": (
            arms["adaptive"]["shock"] == shock_hash == arms["fixed"]["shock"]
        ),
        "fixed_zero_updates": replicate["fixed_weight_update_count"] == 0,
        "fixed_l2_delta_zero": replicate["fixed_parameter_delta_l2"] == 0.0,
        "fixed_fingerprint_unchanged": (
            replicate["fixed_final_fingerprint"] == replicate["frozen_fingerprint"]
            and all(
                episode["policy_fingerprint_start"] == replicate["frozen_fingerprint"]
                and episode["policy_fingerprint_end"] == replicate["frozen_fingerprint"]
                for episode in replicate["fixed_episodes"]
            )
        ),
        "adaptive_exactly_ten_blocks": len(blocks) == 10,
        "adaptive_block_schedule": [b["block_episode"] for b in blocks] == list(range(5, 15)),
        "no_future_data": all(
            list(b["visible_episode_indices"]) == list(range(1, b["block_episode"] + 1))
            for b in blocks
        ),
        "adaptive_update_seeds_match_schedule": [b["update_seed"] for b in blocks]
        == replicate["seeds"]["update"],
        "adaptive_blocks_have_loss_metrics": all(b["loss_metrics"] for b in blocks),
        "adaptive_updates_between_episodes": [
            episode["update_block"] for episode in replicate["adaptive_episodes"]
        ]
        == list(range(5, 15)),
        "fork_fingerprint_identical": replicate["fork_fingerprint"]
        == replicate["frozen_fingerprint"],
        "all_seeds_match_schedule": replicate["seeds"]
        == {phase: list(values) for phase, values in schedule[replicate["training_seed"]].items()},
    }
    return {"all_passed": all(checks.values()), **checks}


@dataclass
class ReplicateResult:
    training_seed: int
    status: str
    schedule_fingerprint: Optional[str] = None
    training_provenance: dict[str, Any] = field(default_factory=dict)
    frozen_fingerprint: Optional[str] = None
    fork_fingerprint: Optional[str] = None
    fixed_final_fingerprint: Optional[str] = None
    pre_shift_performance: Optional[float] = None
    shock_performance: Optional[float] = None
    shared_pre_shift_episodes: list[EpisodeRecord] = field(default_factory=list)
    shared_shock_episodes: list[EpisodeRecord] = field(default_factory=list)
    adaptive_episodes: list[EpisodeRecord] = field(default_factory=list)
    fixed_episodes: list[EpisodeRecord] = field(default_factory=list)
    update_blocks: list[dict[str, Any]] = field(default_factory=list)
    adaptive_recovery: Optional[dict[str, Any]] = None
    fixed_recovery: Optional[dict[str, Any]] = None
    seeds: dict[str, list[int]] = field(default_factory=dict)
    arm_input_fingerprints: dict[str, dict[str, str]] = field(default_factory=dict)
    fixed_weight_update_count: int = 0
    fixed_parameter_delta_l2: float = 0.0
    effective_nominal_parameters: dict[str, Any] = field(default_factory=dict)
    effective_shift_parameters: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _restore_replicate(data: dict[str, Any]) -> ReplicateResult:
    """Restore only records already accepted by the checkpoint digest check."""
    episode_fields = (
        "shared_pre_shift_episodes",
        "shared_shock_episodes",
        "adaptive_episodes",
        "fixed_episodes",
    )
    restored = dict(data)
    for field_name in episode_fields:
        episodes = []
        for episode_data in restored.get(field_name, []):
            episode = dict(episode_data)
            episode["transitions"] = tuple(
                Transition(**transition) for transition in episode.get("transitions", [])
            )
            episodes.append(EpisodeRecord(**episode))
        restored[field_name] = episodes
    return ReplicateResult(**restored)


def _resume_replicate_checkpoint(
    checkpoint_path: Path,
    *,
    study_hash: str,
    protocol_hash: str,
    training_seed: int,
) -> ReplicateResult:
    """Restore one authenticated terminal replicate or fail the whole resume."""
    try:
        checkpoint = read_replicate_checkpoint(
            checkpoint_path,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
            training_seed=training_seed,
        )
        return _restore_replicate(checkpoint)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"resume rejected invalid checkpoint for seed {training_seed}: {exc}"
        ) from exc


def _load_resume_replicates(
    state_root: Path,
    *,
    study_hash: str,
    protocol_hash: str,
    training_seeds: Sequence[int],
) -> dict[int, ReplicateResult]:
    """Validate every stored checkpoint before resuming any unfinished seed."""
    if state_root.is_symlink():
        raise ValueError("resume replicate state path is not a regular directory")
    if not state_root.exists():
        return {}
    if not state_root.is_dir():
        raise ValueError("resume replicate state path is not a regular directory")
    allowed_names = {
        name
        for seed in training_seeds
        for name in (f"seed_{seed}.json", f"seed_{seed}.json.sha256")
    }
    for path in state_root.iterdir():
        if path.name not in allowed_names or path.is_symlink() or not path.is_file():
            raise ValueError(f"resume replicate state contains an unexpected entry: {path.name}")

    restored: dict[int, ReplicateResult] = {}
    for seed in training_seeds:
        checkpoint_path = state_root / f"seed_{seed}.json"
        digest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        if checkpoint_path.exists() != digest_path.exists():
            raise ValueError(f"resume checkpoint or digest is missing for seed {seed}")
        if checkpoint_path.exists():
            restored[seed] = _resume_replicate_checkpoint(
                checkpoint_path,
                study_hash=study_hash,
                protocol_hash=protocol_hash,
                training_seed=seed,
            )
    return restored


def _read_completed_study_artifact(
    artifact_path: Path,
    manifest_path: Path,
    *,
    run_id: str,
    study_hash: str,
) -> dict[str, Any]:
    """Return the persisted result only after validating its immutable envelope."""
    validate_study_manifest(manifest_path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("completed study artifact is malformed") from exc
    artifact_paths = artifact.get("artifact_paths") if isinstance(artifact, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != STUDY_ARTIFACT_SCHEMA_VERSION
        or artifact.get("run_id") != run_id
        or artifact.get("study_hash") != study_hash
        or artifact.get("run_status") not in {"COMPLETE", "PARTIAL"}
        or artifact_paths
        != {
            "json": "adaptive_vs_fixed.json",
            "csv": "adaptive_vs_fixed.csv",
            "manifest": "manifest.json",
        }
    ):
        raise ValueError("completed study artifact has an invalid schema or study identity")
    return artifact


def _run_replicate(
    config: ExperimentConfig,
    training_seed: int,
    training_dir: Path,
    *,
    schedule: dict[int, dict[str, list[int]]],
    smoke: bool,
    trainer_factory: TrainerFactory,
    environment_factory: EnvironmentFactory,
) -> ReplicateResult:
    result = ReplicateResult(training_seed=training_seed, status="failed")
    try:
        benchmark = config.adaptation_benchmark
        if benchmark is None:
            raise ValueError("configuration lacks an Issue #265 adaptation cell")
        result.schedule_fingerprint = schedule_fingerprint(schedule)
        algorithm, training_provenance = _train_once(
            config,
            training_seed,
            training_dir,
            trainer_factory=trainer_factory,
            environment_factory=environment_factory,
            smoke=smoke,
        )
        result.training_provenance = training_provenance
        frozen_fingerprint = model_fingerprint(algorithm)
        result.frozen_fingerprint = frozen_fingerprint
        algorithm.model.policy.set_training_mode(False)

        nominal_params = dict(config.environment.parameters)
        nominal_params["max_steps"] = 8 if smoke else config.environment.max_steps
        shifted_params = dict(nominal_params)
        shifted_params.update(benchmark.shift_parameters)
        result.effective_nominal_parameters = dict(nominal_params)
        result.effective_shift_parameters = dict(shifted_params)
        result.seeds = {phase: list(values) for phase, values in schedule[training_seed].items()}

        nominal_env = _new_env(
            config,
            shifted=False,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            result.shared_pre_shift_episodes = _run_evaluation_segment(
                algorithm=algorithm,
                env=nominal_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="pre",
                indices=range(1, K_PRE + 1),
                seeds=schedule[training_seed]["pre"],
                deterministic=config.evaluation.deterministic,
                arm="shared",
            )
        finally:
            nominal_env.close()
        if model_fingerprint(algorithm) != frozen_fingerprint:
            raise RuntimeError("shared pre-shift evaluation mutated the frozen policy")
        result.pre_shift_performance = float(
            np.mean([episode.reward for episode in result.shared_pre_shift_episodes])
        )

        # Shift introduction occurs only after training and the one shared pre segment.
        shift_before = model_fingerprint(algorithm)
        shock_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            get_effective = getattr(shock_env, "get_effective_parameters", None)
            effective = dict(get_effective()) if callable(get_effective) else shifted_params
            for key, expected in benchmark.shift_parameters.items():
                actual = effective.get(key)
                numeric_match = (
                    isinstance(expected, (int, float))
                    and not isinstance(expected, bool)
                    and isinstance(actual, (int, float))
                    and np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
                )
                if actual != expected and not numeric_match:
                    raise RuntimeError(
                        f"TEST-B parameter {key!r} did not apply: expected {expected!r}, "
                        f"got {actual!r}"
                    )
            result.effective_shift_parameters = effective
            result.shared_shock_episodes = _run_evaluation_segment(
                algorithm=algorithm,
                env=shock_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="post",
                indices=range(1, 6),
                seeds=schedule[training_seed]["post"][:5],
                deterministic=config.evaluation.deterministic,
                arm="shared",
            )
        finally:
            shock_env.close()
        if model_fingerprint(algorithm) != shift_before:
            raise RuntimeError("shift introduction or shared shock evaluation mutated the policy")
        result.shock_performance = float(
            np.mean([episode.reward for episode in result.shared_shock_episodes])
        )

        adaptive, fixed, fork_fingerprint = fork_adaptive_and_fixed(algorithm)
        result.fork_fingerprint = fork_fingerprint
        if fork_fingerprint != frozen_fingerprint:
            raise RuntimeError("Adaptive/Fixed forks did not originate at the frozen fingerprint")
        adapter = _adapter_for(config.algorithm.name.lower())
        post_history = [episode.post_shift_data() for episode in result.shared_shock_episodes]
        adaptive_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            for episode_index in range(6, N_POST + 1):
                boundary = episode_index - 1
                batch = build_update_batch(training_seed, post_history, block_episode=boundary)
                update_log = run_adaptation_update(adaptive, adapter, batch)
                result.update_blocks.append(update_log.to_dict())
                record = _run_evaluation_segment(
                    algorithm=adaptive,
                    env=adaptive_env,
                    training_seed=training_seed,
                    algorithm_name=config.algorithm.name.lower(),
                    environment_name=config.environment.name,
                    phase="post",
                    indices=[episode_index],
                    seeds=[schedule[training_seed]["post"][episode_index - 1]],
                    deterministic=config.evaluation.deterministic,
                    arm="adaptive",
                    initial_update_log=update_log,
                )[0]
                result.adaptive_episodes.append(record)
                post_history.append(record.post_shift_data())
        finally:
            adaptive_env.close()
        validate_block_sequence([block["block_episode"] for block in result.update_blocks])

        fixed_env = _new_env(
            config,
            shifted=True,
            environment_factory=environment_factory,
            max_steps=8 if smoke else None,
        )
        try:
            result.fixed_episodes = _run_evaluation_segment(
                algorithm=fixed,
                env=fixed_env,
                training_seed=training_seed,
                algorithm_name=config.algorithm.name.lower(),
                environment_name=config.environment.name,
                phase="post",
                indices=range(6, N_POST + 1),
                seeds=schedule[training_seed]["post"][5:],
                deterministic=config.evaluation.deterministic,
                arm="fixed",
            )
        finally:
            fixed_env.close()
        result.fixed_final_fingerprint = fixed.fingerprint
        if result.fixed_final_fingerprint != frozen_fingerprint:
            raise RuntimeError("Fixed policy fingerprint changed during its evaluation arm")

        adaptive_post = result.shared_shock_episodes + result.adaptive_episodes
        fixed_post = result.shared_shock_episodes + result.fixed_episodes
        result.adaptive_recovery = _recover(result.shared_pre_shift_episodes, adaptive_post)
        result.fixed_recovery = _recover(result.shared_pre_shift_episodes, fixed_post)
        pre_hash = _return_vector_fingerprint(
            [{"reward": episode.reward} for episode in result.shared_pre_shift_episodes]
        )
        shock_hash = _return_vector_fingerprint(
            [{"reward": episode.reward} for episode in result.shared_shock_episodes]
        )
        result.arm_input_fingerprints = {
            arm: {"pre": pre_hash, "shock": shock_hash} for arm in ("adaptive", "fixed")
        }
        result.status = "completed"
    except Exception as exc:
        result.failure_reason = f"{type(exc).__name__}: {exc}"
        logger.exception("Issue #265 replicate failed for training seed %s", training_seed)
    return result


def run_adaptation_benchmark(
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    training_seeds: Optional[Sequence[int]] = None,
    smoke: bool = False,
    trainer_factory: TrainerFactory = get_trainer,
    environment_factory: EnvironmentFactory = make_env,
    config_path: str | Path | None = None,
    study_run_id: str | None = None,
    study_version: str = "prereg-v1",
    resume: bool = False,
) -> dict[str, Any]:
    """Run the selected preregistered replicates and write JSON/CSV artifacts.

    A smoke run is visibly marked and uses one preregistered seed, a tiny
    training budget, and short episodes. It validates execution only; its
    outputs are not empirical research data.
    """
    if config.adaptation_benchmark is None:
        raise ValueError("configuration must include the Issue #265 adaptation_benchmark section")
    benchmark = config.adaptation_benchmark
    if config.algorithm.name.strip().lower() not in {"ppo", "sac"}:
        raise ValueError("Issue #265 supports only PPO and SAC")
    if config.training is None:
        raise ValueError("Issue #265 requires a training section")
    card = _card_path()
    if not card.is_file():
        raise FileNotFoundError(f"Treatment Card is required before experiment execution: {card}")
    card_sha = _sha256_file(card)
    repository_root = Path.cwd().resolve()
    config_arg = (
        Path(config_path).as_posix()
        if config_path is not None
        else "configs/drone_distribution_shift.yaml"
    )
    schedule = frozen_schedule()
    schedule_fp = schedule_fingerprint(schedule)
    if training_seeds is None:
        selected_seeds = [TRAINING_SEEDS[0]] if smoke else list(TRAINING_SEEDS)
    else:
        selected_seeds = [int(seed) for seed in training_seeds]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("training_seeds must be non-empty and unique")
    if not set(selected_seeds).issubset(TRAINING_SEEDS):
        raise ValueError("all selected training seeds must come from TRAINING_SEEDS")
    if smoke and len(selected_seeds) != 1:
        raise ValueError("smoke mode runs exactly one preregistered training seed")
    if study_run_id is not None:
        if study_version not in {"prereg-v1", "prereg-v2"}:
            raise ValueError(f"unsupported preregistration version: {study_version}")
        if not study_run_id or Path(study_run_id).name != study_run_id:
            raise ValueError("study_run_id must be a non-empty filename-safe component")
        if smoke or selected_seeds != list(TRAINING_SEEDS):
            raise ValueError(
                f"{study_version} requires one non-smoke attempt of all ten seeds in order"
            )
        if output_dir is not None and Path(output_dir).is_absolute():
            raise ValueError(f"{study_version} artifact output_dir must be repository-relative")
        selected_output_dir = (
            Path(output_dir) if output_dir is not None else Path(config.output_dir)
        )
        try:
            selected_output_dir.resolve().relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"{study_version} artifact output_dir must stay inside the repository"
            ) from exc
        if Path(config_arg).is_absolute():
            raise ValueError(f"{study_version} config path must be repository-relative")
        try:
            (repository_root / config_arg).resolve().relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"{study_version} config path must stay inside the repository") from exc
        if config.algorithm.name.strip().lower() != "ppo":
            raise ValueError(f"{study_version} is frozen to the drone_disturbed/ppo cell")
        if config.environment.name != "drone_disturbed" or benchmark.scenario != "TEST-B":
            raise ValueError(f"{study_version} is frozen to drone_disturbed under TEST-B")
        expected_config_hash = (
            ISSUE271_CONFIG_SHA256 if study_version == "prereg-v1" else ISSUE271_V2_CONFIG_SHA256
        )
        if not expected_config_hash or compute_config_sha256(config) != expected_config_hash:
            raise ValueError(f"{study_version} config differs from its frozen Issue #271 configuration")
        if card_sha != ISSUE271_TREATMENT_CARD_SHA256:
            raise ValueError(f"{study_version} Treatment Card differs from the frozen treatment")
    if (
        config.algorithm.name.strip().lower() == "ppo"
        and config.evaluation.deterministic
    ):
        raise ValueError(
            "PPO adaptation requires stochastic behavior-policy action sampling; "
            "deterministic mean actions are not valid on-policy rollout data"
        )
    if study_run_id is not None:
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"cannot verify clean working tree before {study_version} execution"
            ) from exc
        if dirty:
            raise RuntimeError(f"{study_version} execution requires a clean, committed working tree")
    determinism = _enable_study_determinism() if study_run_id is not None else None

    config_file = Path(config_path).resolve() if config_path is not None else None
    config_file_sha = _sha256_file(config_file) if config_file is not None else None

    target_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    if study_run_id is not None:
        target_dir = target_dir / study_run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = "adaptive_vs_fixed" if study_run_id is not None else "adaptation"
    suffixes = (f"{stem}.json", f"{stem}.csv")
    study_hash: str | None = None
    protocol_hash: str | None = None
    if study_run_id is not None:
        runtime_identity = _repository_metadata()
        study_inputs = {
            "study": f"adaptive-vs-fixed/{study_version}",
            "artifact_schema_version": STUDY_ARTIFACT_SCHEMA_VERSION,
            "manifest_schema_version": STUDY_MANIFEST_SCHEMA_VERSION,
            "replicate_checkpoint_schema_version": REPLICATE_CHECKPOINT_SCHEMA_VERSION,
            "study_config": config.model_dump(mode="json", exclude={"output_dir", "log_dir"}),
            "canonical_config_sha256": compute_config_sha256(config),
            "treatment_card_sha256": card_sha,
            "protocol_version": benchmark.protocol_version,
            "protocol_constants": {
                "pre_episodes": K_PRE,
                "post_episodes": N_POST,
                "training_seeds": list(TRAINING_SEEDS),
                "schedule": schedule,
                "schedule_fingerprint": schedule_fp,
            },
            "source_identity": runtime_identity,
            "runtime_determinism": determinism,
        }
        study_hash = write_or_verify_study_manifest(
            study_inputs, target_dir / "study_manifest.json", resume=resume
        )
        protocol_hash = hashlib.sha256(
            canonical_json_bytes(
                {
                    "protocol_version": benchmark.protocol_version,
                    "treatment_card_sha256": card_sha,
                    "protocol_constants": study_inputs["protocol_constants"],
                }
            )
        ).hexdigest()
        final_paths = [target_dir / suffix for suffix in (*suffixes, "manifest.json")]
        if resume and all(path.is_file() for path in final_paths):
            return _read_completed_study_artifact(
                final_paths[0],
                final_paths[-1],
                run_id=study_run_id,
                study_hash=study_hash,
            )
        for path in final_paths:
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact or partial completion: {path}"
                )
    else:
        for suffix in (*suffixes, "manifest.json"):
            if (target_dir / suffix).exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {target_dir / suffix}"
                )
    training_root = target_dir / "training"
    state_root = target_dir / "replicate_state"
    if study_run_id is not None and state_root.exists() and not resume:
        raise FileExistsError("run state already exists; pass --resume to use hashed replicates")
    resumed_replicates: dict[int, ReplicateResult] = {}
    if study_run_id is not None and resume:
        assert study_hash is not None and protocol_hash is not None
        resumed_replicates = _load_resume_replicates(
            state_root,
            study_hash=study_hash,
            protocol_hash=protocol_hash,
            training_seeds=selected_seeds,
        )
    if study_run_id is None:
        for seed in selected_seeds:
            if (training_root / f"seed_{seed}").exists():
                raise FileExistsError(f"refusing to overwrite training output for seed {seed}")

    results: list[ReplicateResult] = []
    for seed in selected_seeds:
        training_dir = training_root / f"seed_{seed}"
        if seed in resumed_replicates:
            results.append(resumed_replicates[seed])
            continue
        if study_run_id is not None and resume and training_dir.exists():
            results.append(
                ReplicateResult(
                    training_seed=seed,
                    status="failed",
                    failure_reason=(
                        "interrupted replicate has no complete hashed checkpoint; "
                        "partial training state was not trusted"
                    ),
                )
            )
            continue
        replicate = _run_replicate(
            config,
            seed,
            training_dir,
            schedule=schedule,
            smoke=smoke,
            trainer_factory=trainer_factory,
            environment_factory=environment_factory,
        )
        results.append(replicate)
        if study_run_id is not None:
            assert study_hash is not None and protocol_hash is not None
            write_replicate_checkpoint(
                replicate.to_dict(),
                state_root / f"seed_{seed}.json",
                study_hash=study_hash,
                protocol_hash=protocol_hash,
                artifact_root=target_dir,
                artifact_directories=(training_dir,),
            )

    vectors: dict[str, tuple[list[Optional[float]], list[Optional[float]]]] = {}
    for cell in PRIMARY_CELLS:
        vectors[cell] = ([None] * len(TRAINING_SEEDS), [None] * len(TRAINING_SEEDS))
    cell_name = f"drone_disturbed/{config.algorithm.name.strip().lower()}"
    if cell_name not in vectors:
        raise ValueError(f"configured Issue #265 cell {cell_name!r} is not preregistered")
    fixed_vector, adaptive_vector = vectors[cell_name]
    for replicate in results:
        if replicate.status != "completed":
            continue
        position = TRAINING_SEEDS.index(replicate.training_seed)
        assert replicate.fixed_recovery is not None
        assert replicate.adaptive_recovery is not None
        fixed_vector[position] = float(replicate.fixed_recovery["truncated_recovery_time"])
        adaptive_vector[position] = float(replicate.adaptive_recovery["truncated_recovery_time"])
    analysis_vectors: dict[str, tuple[Sequence[Optional[float]], Sequence[Optional[float]]]] = {
        cell: (fixed, adaptive) for cell, (fixed, adaptive) in vectors.items()
    }
    paired = analyze_primary_cells(analysis_vectors)
    family_decision = decide_family(
        {cell: analysis.primary_p_value for cell, analysis in paired.items()}
    )

    provenance = _repository_metadata()
    run_status = "PARTIAL" if any(rep.status != "completed" for rep in results) else "COMPLETE"

    def outcome_count(arm: str, status: str) -> int:
        count = 0
        for replicate_result in results:
            if replicate_result.status != "completed":
                continue
            recovery = (
                replicate_result.adaptive_recovery
                if arm == "adaptive"
                else replicate_result.fixed_recovery
            )
            if recovery is not None and recovery.get("status") == status:
                count += 1
        return count

    outcome_summary = {
        arm: {
            **{
                status: outcome_count(arm, status)
                for status in (
                    "recovered",
                    "right_censored",
                    "no_degradation",
                    "degradation_below_resolution",
                )
            },
            "failed_replicates": sum(rep.status != "completed" for rep in results),
        }
        for arm in ("adaptive", "fixed")
    }
    artifact: dict[str, Any] = {
        "schema_version": STUDY_ARTIFACT_SCHEMA_VERSION,
        "run_id": study_run_id,
        "study_hash": study_hash,
        "run_status": run_status if study_run_id is not None else None,
        "protocol_version": benchmark.protocol_version,
        "issue": "271" if study_run_id is not None else "265",
        "run_type": (
            "smoke"
            if smoke
            else study_version
            if study_run_id is not None
            else "full_or_selected_research_run"
        ),
        "determinism": determinism,
        "treatment_card_sha256": card_sha,
        "schedule_fingerprint": schedule_fp,
        "experiment": {
            "name": config.name,
            "algorithm": config.algorithm.name.strip().lower(),
            "environment": config.environment.name,
            "scenario": benchmark.scenario,
            "planned_replicates": len(TRAINING_SEEDS),
            "selected_training_seeds": selected_seeds,
            "config_sha256": compute_config_sha256(config),
            "config": config.model_dump(mode="json"),
        },
        "replicates": [replicate.to_dict() for replicate in results],
        "paired_analysis": {cell: result.to_dict() for cell, result in paired.items()},
        "family_decision": family_decision,
        "failure_summary": {
            "failed_replicates": [
                {"training_seed": replicate.training_seed, "reason": replicate.failure_reason}
                for replicate in results
                if replicate.status != "completed"
            ],
            "completed_replicates": sum(rep.status == "completed" for rep in results),
            "valid_pairs": {cell: result.valid_n for cell, result in paired.items()},
        },
        "outcome_summary": outcome_summary,
        "provenance": {
            **provenance,
            "treatment_card_path": str(card),
            "config_path": str(config_file) if config_file else None,
            "config_file_sha256": config_file_sha,
        },
        "scientific_claim": "Harness execution alone does not establish empirical superiority.",
    }
    for replicate in artifact["replicates"]:
        if replicate["status"] != "completed":
            replicate["invariants"] = {"all_passed": False, "failed_replicate": True}
            continue
        replicate["invariants"] = _audit_replicate_invariants(replicate, schedule)
        if not replicate["invariants"]["all_passed"]:
            raise RuntimeError(
                f"runtime invariant failed for seed {replicate['training_seed']}: "
                f"{replicate['invariants']}"
            )
    artifact["run_status"] = (
        "PARTIAL"
        if study_run_id is not None
        and any(rep["status"] != "completed" for rep in artifact["replicates"])
        else run_status
    )
    if study_run_id is not None:
        artifact["artifact_paths"] = {
            "json": f"{stem}.json",
            "csv": f"{stem}.csv",
            "manifest": "manifest.json",
        }
    if study_run_id is not None:
        json_path, csv_path = write_adaptive_vs_fixed_artifacts(artifact, target_dir)
    else:
        json_path, csv_path = write_adaptation_artifacts(artifact, target_dir, stem=stem)
    if study_run_id is not None:
        output_arg = f"--output-dir {Path(output_dir)} " if output_dir is not None else ""
        executable = Path(sys.argv[0])
        try:
            executable_arg = executable.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (OSError, ValueError):
            executable_arg = executable.name
        command = (
            f"{executable_arg} benchmark adaptation "
            f"--config {config_arg} {output_arg}"
            f"--study {study_version} --run-id {study_run_id}"
        )
        write_study_manifest(
            json_path,
            csv_path,
            target_dir / "manifest.json",
            run_id=study_run_id,
            command=command,
            study_version=study_version,
        )
        assert study_hash is not None
        return _read_completed_study_artifact(
            json_path,
            target_dir / "manifest.json",
            run_id=study_run_id,
            study_hash=study_hash,
        )
    artifact["artifact_paths"] = {"json": str(json_path), "csv": str(csv_path)}
    return artifact


__all__ = ["ReplicateResult", "run_adaptation_benchmark"]
