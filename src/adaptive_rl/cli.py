"""Command Line Interface for AdaptiveRL.

Focuses on the core drone reinforcement learning story:
Train a PPO agent to navigate a simulated 3D drone through obstacles toward
a target, evaluate the trained agent, and visualize the flight demonstration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import adaptive_rl
from adaptive_rl.config import ConfigError, load_config
from adaptive_rl.environments.registry import RegistryError, make_env

app = typer.Typer(
    name="adaptive-rl",
    help="AdaptiveRL: Simulated 3D Drone Reinforcement Learning CLI.",
    add_completion=False,
    no_args_is_help=True,
)

benchmark_app = typer.Typer(
    name="benchmark",
    help="Benchmark commands for learning curves and online adaptation.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")

config_app = typer.Typer(
    name="config",
    help="Configuration inspection and validation commands.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

env_app = typer.Typer(
    name="env",
    help="Environment discovery, inspection, and simulation commands.",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")

experiment_app = typer.Typer(
    name="experiment",
    help="Immutable experiment package validation commands.",
    no_args_is_help=True,
)
app.add_typer(experiment_app, name="experiment")

console = Console()


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


@experiment_app.command(name="validate")
def validate_experiment(
    package_dir: Path = typer.Argument(..., help="Immutable study result package directory"),
    certificate: Optional[Path] = typer.Option(
        None, "--certificate", help="Write validation report outside the package (must not exist)"
    ),
) -> None:
    """Validate package integrity and preregistered claim eligibility."""
    from adaptive_rl.experiments.validator import validate_result_package

    report = validate_result_package(package_dir, certificate_path=certificate)
    for check in report["checks"]:
        color = "green" if check["status"] == "PASS" else "red"
        console.print(f"[{color}]{check['status']}[/{color}] {check['id']}: {check['message']}")
    console.print(
        f"[bold {'green' if report['verdict'] == 'PASS' else 'red'}]Verdict: {report['verdict']}[/bold {'green' if report['verdict'] == 'PASS' else 'red'}]"
    )
    if report["verdict"] != "PASS":
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show the installed AdaptiveRL version and project story."""
    console.print(
        f"[bold green]AdaptiveRL[/bold green] version [bold cyan]{adaptive_rl.__version__}[/bold cyan]\n"
        "[italic]Autonomous 3D Drone Navigation via Reinforcement Learning[/italic]"
    )


@config_app.command(name="validate")
def validate_config(
    path: Path = typer.Argument(..., help="Path to YAML configuration file to validate"),
) -> None:
    """Validate an experiment YAML configuration file against the schema."""
    try:
        cfg = load_config(path)
        training_info = (
            f"{cfg.training.total_timesteps:,} steps (checkpoint freq: {cfg.training.checkpoint_freq})"
            if cfg.training is not None
            else "None"
        )
        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration is valid![/bold green]\n\n"
                f"• [bold]Experiment:[/bold] {cfg.name}\n"
                f"• [bold]Seed:[/bold] {cfg.seed}\n"
                f"• [bold]Algorithm:[/bold] {cfg.algorithm.name.upper()} (LR: {cfg.algorithm.learning_rate}, Gamma: {cfg.algorithm.gamma})\n"
                f"• [bold]Environment:[/bold] {cfg.environment.name} (Max steps: {cfg.environment.max_steps})\n"
                f"• [bold]Training:[/bold] {training_info}\n"
                f"• [bold]Evaluation:[/bold] {cfg.evaluation.eval_episodes} episodes",
                title=f"Valid Configuration: {path.name}",
                border_style="green",
            )
        )
    except ConfigError as err:
        console.print(
            Panel.fit(
                f"[bold red]Configuration validation error:[/bold red]\n\n{err}",
                title=f"Invalid: {path}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@env_app.command(name="inspect")
def inspect_env(
    name: str = typer.Argument("drone", help="Name of registered environment to inspect"),
) -> None:
    """Inspect observation and action spaces of an environment."""
    try:
        env = make_env(name)
        obs, info = env.reset(seed=42)

        panel_content = [
            f"[bold green]Environment '{name}' verified successfully![/bold green]\n",
            f"• [bold]Type:[/bold] {type(env).__name__}",
            f"• [bold]Observation Space:[/bold] {env.observation_space}",
            f"• [bold]Action Space:[/bold] {env.action_space}",
            f"• [bold]Initial Observation Shape:[/bold] {getattr(obs, 'shape', 'unknown')}",
            f"• [bold]Reset Info:[/bold] {info}",
        ]

        if hasattr(env, "render"):
            rendered = env.render()
            if rendered:
                panel_content.append(f"\n[bold]Initial Layout:[/bold]\n{rendered}")

        env.close()

        console.print(
            Panel.fit(
                "\n".join(panel_content),
                title=f"Environment Inspection: {name}",
                border_style="cyan",
            )
        )
    except RegistryError as err:
        console.print(
            Panel.fit(
                f"[bold red]Environment inspection failed:[/bold red]\n\n{err}",
                title=f"Error: {name}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to training configuration YAML"
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Override total training timesteps"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Override experiment random seed"
    ),
    split: Optional[str] = typer.Option(
        None, "--split", help="Environment dataset split ('train' or 'test')"
    ),
) -> None:
    """Train a reinforcement learning agent using PPO."""
    if config is None:
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break
        if config is None:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if exp_config.training is None:
        console.print("[bold red]Configuration error:[/bold red] 'training' section is required.")
        raise typer.Exit(code=1)

    if timesteps is not None:
        exp_config.training.total_timesteps = timesteps
    if seed is not None:
        exp_config.seed = seed

    clean_split: Optional[str] = None
    if split is not None:
        clean_split = split.strip().lower()
        if clean_split not in ("train", "test"):
            console.print(
                f"[bold red]Invalid split:[/bold red] '{split}'. Expected 'train' or 'test'."
            )
            raise typer.Exit(code=1)
        exp_config.environment.parameters["split"] = clean_split

    split_info = f"\n• [bold]Split:[/bold] {clean_split}" if clean_split is not None else ""
    console.print(
        Panel.fit(
            f"[bold green]Starting Drone RL Training: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name.upper()}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Total Timesteps:[/bold] {exp_config.training.total_timesteps:,}\n"
            f"• [bold]Checkpoint Freq:[/bold] {exp_config.training.checkpoint_freq}\n"
            f"• [bold]Seed:[/bold] {exp_config.seed}\n"
            f"• [bold]Output Dir:[/bold] {exp_config.output_dir}"
            f"{split_info}",
            title="PPO Drone Training Pipeline",
            border_style="cyan",
        )
    )

    from adaptive_rl.training.trainer import get_trainer

    try:
        trainer = get_trainer(config=exp_config)
        result = trainer.fit()

        console.print(
            Panel.fit(
                f"[bold green]Training Completed Successfully![/bold green]\n\n"
                f"• [bold]Total Timesteps Trained:[/bold] {result.total_timesteps:,}\n"
                f"• [bold]Episodes Completed:[/bold] {result.episodes_completed}\n"
                f"• [bold]Mean Reward (last window):[/bold] {result.mean_reward:.2f}\n"
                f"• [bold]Saved Model:[/bold] {result.final_model_path}\n"
                f"• [bold]Metadata:[/bold] {result.metadata_path}",
                title="Training Summary",
                border_style="green",
            )
        )
    except Exception as err:
        console.print(f"[bold red]Training failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@benchmark_app.command(name="budgets")
def benchmark_budgets(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    budgets: Optional[str] = typer.Option(
        None, "--budgets", help="Comma-separated training budgets (for example: 5000,10000,25000)"
    ),
    training_seed: Optional[int] = typer.Option(
        None, "--training-seed", help="Override the training seed used across budgets"
    ),
    eval_seeds: Optional[str] = typer.Option(
        None,
        "--eval-seeds",
        help="Comma-separated evaluation seed groups; defaults to configured or benchmark seeds",
    ),
    episodes: Optional[int] = typer.Option(
        None, "--episodes", help="Evaluation episodes per seed (separate from the seed count)"
    ),
    deterministic: Optional[bool] = typer.Option(
        None, "--deterministic/--stochastic", help="Use deterministic actions during evaluation"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory for benchmark JSON/CSV/plot artifacts"
    ),
    plot: bool = typer.Option(False, "--plot/--no-plot", help="Render a learning-curve plot"),
    plot_x_axis: str = typer.Option(
        "trained",
        "--plot-x-axis",
        help="Plot x-axis semantics: 'trained' (actual timesteps) or 'requested' (budget)",
    ),
) -> None:
    """Run the PPO learning-curve benchmark across training budgets (PPO only)."""
    if config is None:
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break
        if config is None:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    x_axis = plot_x_axis.strip().lower()
    if x_axis not in ("trained", "requested"):
        console.print(
            f"[bold red]Invalid --plot-x-axis:[/bold red] {plot_x_axis!r}. "
            "Expected 'trained' or 'requested'."
        )
        raise typer.Exit(code=1)

    try:
        from adaptive_rl.benchmarking import (
            BenchmarkRunError,
            run_learning_curve_benchmark,
            validate_budgets,
        )
    except ImportError as err:
        console.print(f"[bold red]Benchmarking module unavailable:[/bold red] {err}")
        raise typer.Exit(code=1)

    try:
        budget_values = validate_budgets(budgets) if budgets is not None else None
        if eval_seeds is not None:
            seed_tokens = [part.strip() for part in eval_seeds.split(",")]
            if not seed_tokens or any(not token for token in seed_tokens):
                raise ValueError(
                    "Malformed evaluation seed list: expected comma-separated integers."
                )
            parsed_eval_seeds = []
            for token in seed_tokens:
                try:
                    parsed_eval_seeds.append(int(token))
                except ValueError as err:
                    raise ValueError(f"Malformed evaluation seed value: {token!r}") from err
        else:
            parsed_eval_seeds = None

        result = run_learning_curve_benchmark(
            exp_config,
            budgets=budget_values,
            training_seed=training_seed,
            evaluation_seeds=parsed_eval_seeds,
            evaluation_episodes=episodes,
            deterministic=deterministic,
            output_dir=output_dir,
            plot=plot,
            plot_x_axis=cast(Literal["trained", "requested"], x_axis),
        )
    except BenchmarkRunError as err:
        partial = err.result
        completed = ", ".join(str(b) for b in partial.completed_budgets) or "none"
        console.print(
            Panel.fit(
                f"[bold red]PPO learning-curve benchmark FAILED[/bold red]\n\n"
                f"• [bold]Status:[/bold] {partial.status}\n"
                f"• [bold]Completed budgets:[/bold] {completed}\n"
                f"• [bold]Failed budget:[/bold] {partial.failed_budget}\n"
                f"• [bold]Error:[/bold] {partial.error}\n"
                f"• [bold]JSON:[/bold] {partial.json_path}\n"
                f"• [bold]CSV:[/bold] {partial.csv_path}",
                title="Learning Curve Benchmark (partial)",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    except Exception as err:
        console.print(f"[bold red]Benchmark failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if result.plot_path is None:
        plot_summary = "not requested"
    elif result.plot_error is not None:
        plot_summary = f"FAILED ({result.plot_error})"
    else:
        plot_summary = str(result.plot_path)

    console.print(
        Panel.fit(
            f"[bold green]PPO learning-curve benchmark complete[/bold green]\n\n"
            f"• [bold]Status:[/bold] {result.status}\n"
            f"• [bold]Budgets:[/bold] {', '.join(str(b) for b in result.budgets)}\n"
            f"• [bold]Completed budgets:[/bold] "
            f"{', '.join(str(b) for b in result.completed_budgets)}\n"
            f"• [bold]Training seed:[/bold] {result.training_seed}\n"
            f"• [bold]Evaluation seeds:[/bold] {result.evaluation_seeds}\n"
            f"• [bold]Episodes per seed:[/bold] {result.evaluation_episodes}\n"
            f"• [bold]JSON:[/bold] {result.json_path}\n"
            f"• [bold]CSV:[/bold] {result.csv_path}\n"
            f"• [bold]Plot:[/bold] {plot_summary}",
            title="Learning Curve Benchmark",
            border_style="green",
        )
    )
    if result.plot_error is not None:
        console.print(
            "[bold red]Plot generation failed; JSON/CSV artifacts were preserved.[/bold red]"
        )
        raise typer.Exit(code=1)


@benchmark_app.command(name="adaptation")
def benchmark_adaptation(
    config: Path = typer.Option(
        Path("configs/drone_distribution_shift.yaml"),
        "--config",
        "-c",
        help="Issue #265 nominal training and TEST-B configuration",
    ),
    algorithm: Optional[str] = typer.Option(
        None, "--algorithm", help="Algorithm cell: ppo or sac (defaults to config value)"
    ),
    training_seeds: Optional[str] = typer.Option(
        None,
        "--training-seeds",
        help="Comma-separated preregistered training seeds; defaults to all ten",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory for Issue #265 JSON/CSV and training artifacts"
    ),
    study: Optional[str] = typer.Option(
        None, "--study", help="Run the immutable full protocol study (currently prereg-v1)"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Unique immutable output directory name required with --study"
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse only complete hashed replicate checkpoints for an unfinished study run",
    ),
    deterministic: Optional[bool] = typer.Option(
        None,
        "--deterministic/--stochastic",
        help="Override evaluation action selection; PPO adaptation requires stochastic actions",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Run one explicitly labeled, reduced-size machinery check (not research data)",
    ),
) -> None:
    """Run the preregistered train-once, forked Adaptive-vs-Fixed experiment."""
    try:
        exp_config = load_config(config)
        if algorithm is not None:
            selected_algorithm = algorithm.strip().lower()
            if selected_algorithm not in {"ppo", "sac"}:
                raise ValueError("--algorithm must be 'ppo' or 'sac'")
            algorithm_config = exp_config.algorithm.model_copy(deep=True)
            if selected_algorithm != algorithm_config.name.strip().lower():
                algorithm_config.name = selected_algorithm
                if selected_algorithm == "sac":
                    algorithm_config.parameters = {
                        "buffer_size": 100_000,
                        "learning_starts": 100,
                        "train_freq": 1,
                        "gradient_steps": 1,
                        "tau": 0.005,
                        "ent_coef": "auto",
                    }
                else:
                    algorithm_config.parameters = {
                        "n_steps": 1024,
                        "n_epochs": 10,
                        "clip_range": 0.2,
                        "ent_coef": 0.01,
                    }
                exp_config = exp_config.model_copy(
                    update={"algorithm": algorithm_config}, deep=True
                )
        if deterministic is not None:
            evaluation_config = exp_config.evaluation.model_copy(deep=True)
            evaluation_config.deterministic = deterministic
            exp_config = exp_config.model_copy(update={"evaluation": evaluation_config}, deep=True)

        selected_seeds = None
        if training_seeds is not None:
            tokens = [token.strip() for token in training_seeds.split(",")]
            if not tokens or any(not token for token in tokens):
                raise ValueError("--training-seeds expects comma-separated integers")
            try:
                selected_seeds = [int(token) for token in tokens]
            except ValueError as exc:
                raise ValueError("--training-seeds expects comma-separated integers") from exc

        from adaptive_rl.benchmarking.adaptation_runner import run_adaptation_benchmark

        if study not in {None, "prereg-v1"}:
            raise ValueError("--study currently supports only prereg-v1")
        if (study is None) != (run_id is None):
            raise ValueError("--study and --run-id must be supplied together")
        if resume and study is None:
            raise ValueError("--resume requires --study prereg-v1 and --run-id")

        artifact = run_adaptation_benchmark(
            exp_config,
            output_dir=output_dir,
            training_seeds=selected_seeds,
            smoke=smoke,
            config_path=config,
            study_run_id=run_id if study is not None else None,
            resume=resume,
        )
    except Exception as err:
        console.print(f"[bold red]Issue #265 benchmark failed:[/bold red] {err}")
        raise typer.Exit(code=1)

    failed = artifact["failure_summary"]["failed_replicates"]
    completed = artifact["failure_summary"]["completed_replicates"]
    json_artifact_path = Path(artifact["artifact_paths"]["json"])
    csv_artifact_path = Path(artifact["artifact_paths"]["csv"])
    if study is not None and run_id is not None:
        study_dir = (output_dir or exp_config.output_dir) / run_id
        json_artifact_path = study_dir / json_artifact_path
        csv_artifact_path = study_dir / csv_artifact_path
    console.print(
        Panel.fit(
            f"[bold]{'Smoke check' if smoke else 'Issue #265 benchmark'} finished[/bold]\n\n"
            f"• [bold]Run type:[/bold] {artifact['run_type']}\n"
            f"• [bold]Algorithm:[/bold] {artifact['experiment']['algorithm']}\n"
            f"• [bold]Completed replicates:[/bold] {completed}\n"
            f"• [bold]Failed replicates:[/bold] {len(failed)}\n"
            f"• [bold]JSON:[/bold] {json_artifact_path}\n"
            f"• [bold]CSV:[/bold] {csv_artifact_path}\n"
            f"• [bold]Scientific result:[/bold] not established by harness execution",
            title="Online Adaptation Benchmark",
            border_style="yellow" if smoke or failed else "green",
        )
    )
    if failed:
        raise typer.Exit(code=1)


@app.command(context_settings={"allow_extra_args": True})
def evaluate(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to trained model weights (.zip)"
    ),
    episodes: Optional[int] = typer.Option(
        20, "--episodes", "-e", help="Number of evaluation episodes"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Base seed for single-seed evaluation (defaults to config seed)"
    ),
    seeds: Optional[List[int]] = typer.Option(
        None, "--seeds", help="Explicit evaluation seeds; each seed runs --episodes episodes"
    ),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic action selection"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Optional path to export JSON metrics report"
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--output-csv", help="Optional path to export aggregated multi-seed CSV"
    ),
    compare_random: bool = typer.Option(
        False,
        "--compare-random",
        help="Compare PPO against random action baseline under identical conditions",
    ),
    split: Optional[str] = typer.Option(
        None, "--split", help="Environment dataset split ('train' or 'test')"
    ),
) -> None:
    """Evaluate a trained agent over multiple benchmark episodes."""
    if ctx.args:
        if seeds is None:
            console.print(
                "[bold red]Unexpected positional values; provide seeds with --seeds.[/bold red]"
            )
            raise typer.Exit(code=1)
        try:
            seeds.extend(int(value) for value in ctx.args)
        except ValueError:
            console.print(
                "[bold red]Seeds must be integers; use --seeds followed by space-separated values.[/bold red]"
            )
            raise typer.Exit(code=1)

    if config is None:
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break
        if config is None:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    num_episodes = episodes if episodes is not None else exp_config.evaluation.eval_episodes
    if num_episodes <= 0:
        console.print("[bold red]Evaluation episodes must be positive.[/bold red]")
        raise typer.Exit(code=1)
    if seed is not None and seed < 0:
        console.print("[bold red]Evaluation seed must be non-negative.[/bold red]")
        raise typer.Exit(code=1)
    if seeds is not None and seed is not None:
        console.print("[bold red]Use either --seed or --seeds, not both.[/bold red]")
        raise typer.Exit(code=1)
    if seeds is not None and compare_random:
        console.print("[bold red]--compare-random cannot be combined with --seeds.[/bold red]")
        raise typer.Exit(code=1)

    if split is not None:
        clean_split = split.strip().lower()
        if clean_split not in ("train", "test"):
            console.print(
                f"[bold red]Invalid split:[/bold red] '{split}'. Expected 'train' or 'test'."
            )
            raise typer.Exit(code=1)
        if seeds is not None:
            console.print("[bold red]--split cannot be combined with --seeds.[/bold red]")
            raise typer.Exit(code=1)
        exp_config.environment.parameters["split"] = clean_split
    else:
        clean_split = None

    # Resolve model path
    if model is None:
        candidate = exp_config.output_dir / "models" / f"{exp_config.name}_final.zip"
        if candidate.exists():
            model = candidate
        else:
            console.print(
                f"[bold red]No model weights provided.[/bold red] Pass --model <path> or train first to generate {candidate}"
            )
            raise typer.Exit(code=1)

    split_info = f"\n• [bold]Split:[/bold] {clean_split}" if clean_split is not None else ""
    console.print(
        Panel.fit(
            f"[bold green]Starting Evaluation: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Episodes:[/bold] {num_episodes}\n"
            f"• [bold]Deterministic:[/bold] {deterministic}\n"
            f"• [bold]Seeds:[/bold] {seeds if seeds is not None else seed if seed is not None else exp_config.seed}"
            f"{split_info}",
            title="Evaluation Engine",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.evaluation.evaluator import Evaluator

    try:
        env = make_env(
            exp_config.environment.name,
            **exp_config.environment.parameters,
        )
        algo = PPOAlgorithm.from_pretrained(model, env=env)
        evaluator = Evaluator(algorithm=algo, env=env)

        metrics = None
        if seeds is not None:
            multi_seed_result = evaluator.evaluate_seeds(
                seeds=seeds,
                episodes_per_seed=num_episodes,
                deterministic=deterministic,
            )
            console.print(
                f"\nSeeds: {len(multi_seed_result.seeds)} | "
                f"Episodes per seed: {multi_seed_result.episodes_per_seed} | "
                f"Total episodes: {multi_seed_result.total_episodes}"
            )
            table = Table(title="Multi-Seed Evaluation (95% Student's t CI)")
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")
            table.add_column("95% CI lower", justify="right")
            table.add_column("95% CI upper", justify="right")
            for metric_name, label in (
                ("mean_reward", "Mean reward"),
                ("success_rate", "Success rate"),
                ("collision_rate", "Collision rate"),
                ("mean_episode_length", "Episode length"),
            ):
                summary = multi_seed_result.aggregate[metric_name]
                table.add_row(
                    label,
                    _format_metric(summary.mean),
                    _format_metric(summary.std),
                    _format_metric(summary.ci95_lower),
                    _format_metric(summary.ci95_upper),
                )
            console.print(table)
            report_target = output_report or (exp_config.output_dir / "evaluation_multiseed.json")
            csv_target = output_csv or (exp_config.output_dir / "evaluation_multiseed.csv")
            json_saved, csv_saved = evaluator.save_multiseed_report(
                multi_seed_result, report_target, csv_target
            )
            console.print(f"\n[bold green]JSON report saved to:[/bold green] {json_saved}")
            console.print(f"[bold green]CSV report saved to:[/bold green] {csv_saved}")
        else:
            metrics = evaluator.evaluate(
                num_episodes=num_episodes,
                deterministic=deterministic,
                base_seed=(exp_config.seed if seed is None else seed)
                if clean_split is None
                else None,
                split=clean_split,
            )

            success_pct = (
                f"{metrics.success_rate * 100:.1f}%" if metrics.success_rate is not None else "N/A"
            )
            collision_pct = (
                f"{metrics.collision_rate * 100:.1f}%"
                if metrics.collision_rate is not None
                else "N/A"
            )

            console.print("\n[bold]## Evaluation[/bold]")
            console.print(f"Episodes: {metrics.episodes}")
            console.print(f"Success rate: {success_pct}")
            console.print(f"Collision rate: {collision_pct}")
            console.print(f"Mean reward: {metrics.mean_reward:.2f}")
            console.print(f"Mean episode length: {metrics.mean_episode_length:.1f}\n")

            table = Table(title=f"Benchmark Results ({num_episodes} episodes)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_row("Mean Reward", f"{metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
            table.add_row(
                "Min / Max Reward", f"{metrics.min_reward:.2f} / {metrics.max_reward:.2f}"
            )
            table.add_row("Success Rate", success_pct)
            table.add_row("Collision Rate", collision_pct)
            table.add_row(
                "Mean Episode Length",
                f"{metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}",
            )
            if (
                metrics.obstacle_collision_count is not None
                and metrics.boundary_collision_count is not None
            ):
                table.add_row(
                    "  • Obstacle Collisions",
                    f"{metrics.obstacle_collision_count} ({(metrics.obstacle_collision_rate or 0.0) * 100:.1f}%)",
                )
                table.add_row(
                    "  • Boundary Collisions",
                    f"{metrics.boundary_collision_count} ({(metrics.boundary_collision_rate or 0.0) * 100:.1f}%)",
                )
            if metrics.mean_path_length is not None:
                table.add_row(
                    "Mean Path Length",
                    f"{metrics.mean_path_length:.2f} ± {metrics.std_path_length or 0.0:.2f} m",
                )
            if metrics.mean_straight_line_distance is not None:
                table.add_row(
                    "Straight-Line Distance", f"{metrics.mean_straight_line_distance:.2f} m"
                )
            if metrics.mean_path_efficiency is not None:
                table.add_row("Path Efficiency", f"{metrics.mean_path_efficiency * 100:.1f}%")
            if metrics.mean_min_obstacle_clearance is not None:
                import math

                clearance_str = (
                    f"{metrics.mean_min_obstacle_clearance:.2f} m"
                    if math.isfinite(metrics.mean_min_obstacle_clearance)
                    else "N/A"
                )
                table.add_row("Min Obstacle Clearance", clearance_str)
            if metrics.mean_max_velocity is not None:
                table.add_row("Max Velocity", f"{metrics.mean_max_velocity:.2f} m/s")
            if metrics.mean_max_acceleration is not None:
                table.add_row("Max Acceleration", f"{metrics.mean_max_acceleration:.2f} m/s²")
            console.print(table)

        if compare_random and metrics is not None:
            from adaptive_rl.evaluation.evaluator import compare_policies

            comp_results = compare_policies(
                ppo_algorithm=algo,
                env=env,
                num_episodes=num_episodes,
                base_seed=(exp_config.seed if seed is None else seed)
                if clean_split is None
                else None,
                split=clean_split,
            )
            comp_table = Table(title=f"Policy Comparison ({num_episodes} episodes)")
            comp_table.add_column("Policy", style="cyan")
            comp_table.add_column("Success Rate", justify="right")
            comp_table.add_column("Collision Rate", justify="right")
            comp_table.add_column("Mean Reward", justify="right")
            comp_table.add_column("Mean Steps", justify="right")

            for pol_name, m in comp_results.items():
                s_pct = f"{m.success_rate * 100:.1f}%" if m.success_rate is not None else "N/A"
                c_pct = f"{m.collision_rate * 100:.1f}%" if m.collision_rate is not None else "N/A"
                comp_table.add_row(
                    pol_name,
                    s_pct,
                    c_pct,
                    f"{m.mean_reward:.2f}",
                    f"{m.mean_episode_length:.1f}",
                )
            console.print("\n")
            console.print(comp_table)

        if metrics is not None:
            report_target = output_report or (exp_config.output_dir / "evaluation.json")
            saved_path = evaluator.save_report(metrics, report_target)
            console.print(f"\n[bold green]Report saved to:[/bold green] {saved_path}")

            csv_target = output_csv or (exp_config.output_dir / "evaluation.csv")
            saved_csv = evaluator.save_csv_report(metrics, csv_target)
            console.print(f"[bold green]CSV report saved to:[/bold green] {saved_csv}")

        env.close()
    except Exception as err:
        console.print(f"[bold red]Evaluation failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command(name="evaluate-generalization")
def evaluate_generalization_cmd(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model weights (.zip)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    episodes: int = typer.Option(
        20, "--episodes", "-e", help="Number of evaluation episodes per split"
    ),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic action selection"
    ),
    output_report: Optional[Path] = typer.Option(
        Path("artifacts/generalization_benchmark.json"),
        "--output-report",
        "-o",
        help="Path to export structured JSON benchmark report",
    ),
) -> None:
    """Evaluate trained policy on train and unseen test distributions and compute generalization gaps."""
    if not model.exists():
        console.print(f"[bold red]Model file does not exist:[/bold red] {model}")
        raise typer.Exit(code=1)

    if episodes <= 0:
        console.print(f"[bold red]Episodes must be positive:[/bold red] got {episodes}")
        raise typer.Exit(code=1)

    if config is None:
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break

    env_name = "drone"
    env_params: Dict[str, Any] = {}
    exp_name = "drone_generalization"
    if config is not None and config.exists():
        try:
            exp_config = load_config(config)
            env_name = exp_config.environment.name
            env_params = exp_config.environment.parameters
            exp_name = exp_config.name
        except ConfigError as err:
            console.print(f"[bold red]Configuration error:[/bold red] {err}")
            raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Running Unseen-Environment Generalization Benchmark[/bold cyan]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Environment:[/bold] {env_name}\n"
            f"• [bold]Episodes per Split:[/bold] {episodes}\n"
            f"• [bold]Deterministic:[/bold] {deterministic}\n"
            f"• [bold]Benchmark Report:[/bold] {output_report}",
            title="Generalization Evaluation",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.evaluation.generalization import evaluate_generalization

    try:
        env = make_env(env_name, **env_params)
        algo = PPOAlgorithm.from_pretrained(model, env=env)

        benchmark_result = evaluate_generalization(
            algorithm=algo,
            env=env,
            num_episodes=episodes,
            deterministic=deterministic,
            output_path=output_report,
            metadata={
                "model": str(model),
                "config": str(config) if config else None,
                "experiment_name": exp_name,
            },
        )

        train_res = benchmark_result.train
        test_res = benchmark_result.test
        gap_res = benchmark_result.generalization_gap

        table = Table(title=f"Generalization Benchmark Results ({episodes} episodes per split)")
        table.add_column("Distribution", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Collision Rate", justify="right")
        table.add_column("Mean Reward", justify="right")
        table.add_column("Mean Steps", justify="right")

        t_succ = (
            f"{train_res['success_rate'] * 100:.1f}%"
            if train_res.get("success_rate") is not None
            else "N/A"
        )
        t_coll = (
            f"{train_res['collision_rate'] * 100:.1f}%"
            if train_res.get("collision_rate") is not None
            else "N/A"
        )
        table.add_row(
            "TRAIN (seen seeds)",
            t_succ,
            t_coll,
            f"{train_res['mean_reward']:.2f}",
            f"{train_res['mean_episode_length']:.1f}",
        )

        e_succ = (
            f"{test_res['success_rate'] * 100:.1f}%"
            if test_res.get("success_rate") is not None
            else "N/A"
        )
        e_coll = (
            f"{test_res['collision_rate'] * 100:.1f}%"
            if test_res.get("collision_rate") is not None
            else "N/A"
        )
        table.add_row(
            "TEST (unseen seeds)",
            e_succ,
            e_coll,
            f"{test_res['mean_reward']:.2f}",
            f"{test_res['mean_episode_length']:.1f}",
        )

        console.print("\n")
        console.print(table)

        gap_table = Table(title="Generalization Gaps (Train − Test)")
        gap_table.add_column("Metric Gap", style="cyan")
        gap_table.add_column("Delta (Train − Test)", style="green", justify="right")

        succ_gap = gap_res.get("success")
        gap_table.add_row(
            "Success Generalization Gap (Δ_success)",
            f"{succ_gap * 100:+.1f}%" if succ_gap is not None else "N/A",
        )
        rew_gap = gap_res.get("reward")
        gap_table.add_row(
            "Reward Generalization Gap (Δ_reward)",
            f"{rew_gap:+.2f}" if rew_gap is not None else "N/A",
        )
        console.print("\n")
        console.print(gap_table)

        if output_report is not None:
            console.print(f"\n[bold green]Benchmark report saved to:[/bold green] {output_report}")

    except Exception as err:
        console.print(f"[bold red]Generalization evaluation failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)
    finally:
        if "env" in locals():
            env.close()


@app.command(name="experiment-density")
def experiment_density(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model weights (.zip)"),
    episodes: int = typer.Option(
        10, "--episodes", "-e", help="Episodes per obstacle density condition"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed"),
    output_report: Optional[Path] = typer.Option(
        Path("artifacts/obstacle_density_experiment.json"),
        "--output-report",
        "-o",
        help="Optional path to export JSON metrics report",
    ),
) -> None:
    """Evaluate a trained agent across varied obstacle densities (4, 6, 8 obstacles)."""
    if not model.exists():
        console.print(f"[bold red]Model file does not exist:[/bold red] {model}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Running Obstacle-Density Experiment[/bold cyan]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Obstacle Densities:[/bold] 4, 6, 8 obstacles\n"
            f"• [bold]Episodes per Condition:[/bold] {episodes}\n"
            f"• [bold]Seed:[/bold] {seed}\n\n"
            f"[dim]Hypothesis: More obstacles increase navigation difficulty.[/dim]",
            title="Obstacle-Density Experiment",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.environments.drone import DroneNavigation3DEnv
    from adaptive_rl.evaluation.evaluator import run_obstacle_density_experiment

    dummy_env = DroneNavigation3DEnv()
    try:
        algo = PPOAlgorithm.from_pretrained(model, env=dummy_env)
        results = run_obstacle_density_experiment(
            algorithm=algo,
            obstacle_counts=(4, 6, 8),
            episodes_per_density=episodes,
            base_seed=seed,
            output_path=output_report,
        )

        table = Table(title="Obstacle-Density Results")
        table.add_column("Obstacles", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Collision Rate", justify="right")
        table.add_column("Mean Reward", justify="right")
        table.add_column("Mean Steps", justify="right")

        for r in results:
            table.add_row(
                f"{r['obstacle_count']} Obstacles",
                f"{r['success_rate'] * 100:.1f}%",
                f"{r['collision_rate'] * 100:.1f}%",
                f"{r['mean_reward']:.2f}",
                f"{r['mean_episode_length']:.1f}",
            )

        console.print("\n")
        console.print(table)
        if output_report:
            console.print(f"\n[bold green]Report saved to:[/bold green] {output_report}")
    except Exception as err:
        console.print(f"[bold red]Experiment failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)
    finally:
        dummy_env.close()


@app.command(name="experiment-ablation")
def experiment_ablation(
    timesteps: int = typer.Option(
        25000,
        "--timesteps",
        "-t",
        help="Training timesteps budget per variant",
    ),
    episodes: int = typer.Option(
        20,
        "--episodes",
        "-e",
        help="Number of held-out evaluation episodes per variant",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Deterministic base seed for training and evaluation",
    ),
    output_report: Optional[Path] = typer.Option(
        None,
        "--output-report",
        "-o",
        help="Optional path to export JSON benchmark report",
    ),
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output-csv",
        help="Optional path to export CSV benchmark report",
    ),
    eval_freq: Optional[int] = typer.Option(
        None,
        "--eval-freq",
        help="Frequency of intermediate evaluations for convergence tracking",
    ),
) -> None:
    """Run controlled reward-function ablation experiments across Variants A-D."""
    console.print(
        Panel.fit(
            f"[bold cyan]Running Reward-Function Ablation Study[/bold cyan]\n\n"
            f"• [bold]Variants:[/bold] A (Progress Only), B (+Collision), C (+Step), D (Full Baseline)\n"
            f"• [bold]Training Budget:[/bold] {timesteps:,} steps/variant\n"
            f"• [bold]Evaluation Episodes:[/bold] {episodes}\n"
            f"• [bold]Base Seed:[/bold] {seed}\n\n"
            f"[dim]Hypothesis: Explicit collision and step penalties improve safe goal-directed flight.[/dim]",
            title="Reward Ablation Benchmark",
            border_style="cyan",
        )
    )

    from adaptive_rl.benchmarking.ablation import run_reward_ablation_experiment

    try:
        report_json = output_report or Path("artifacts/benchmarks/reward_ablation.json")
        report_csv = output_csv or Path("artifacts/benchmarks/reward_ablation.csv")

        data = run_reward_ablation_experiment(
            timesteps=timesteps,
            eval_episodes=episodes,
            seed=seed,
            eval_freq=eval_freq,
            output_json=report_json,
            output_csv=report_csv,
        )

        results = data.get("results", [])

        table = Table(title="Reward-Function Ablation Benchmark Results")
        table.add_column("Variant", style="bold cyan")
        table.add_column("Success", justify="right")
        table.add_column("Collision", justify="right")
        table.add_column("Timeout", justify="right")
        table.add_column("Mean Reward", justify="right")
        table.add_column("Path Efficiency", justify="right")
        table.add_column("Convergence", justify="right")

        for r in results:
            succ_str = (
                f"{r['success_rate'] * 100:.1f}%" if r.get("success_rate") is not None else "N/A"
            )
            coll_str = (
                f"{r['collision_rate'] * 100:.1f}%"
                if r.get("collision_rate") is not None
                else "N/A"
            )
            time_str = (
                f"{r['timeout_rate'] * 100:.1f}%" if r.get("timeout_rate") is not None else "N/A"
            )
            rew_str = f"{r['mean_reward']:.2f}" if r.get("mean_reward") is not None else "N/A"
            eff_str = (
                f"{r['mean_path_efficiency'] * 100:.1f}%"
                if r.get("mean_path_efficiency") is not None
                else "N/A"
            )
            conv_speed = r.get("convergence_speed")
            conv_str = (
                f"{conv_speed:,} steps" if conv_speed is not None else "[dim]not reached[/dim]"
            )

            table.add_row(
                r["variant"],
                succ_str,
                coll_str,
                time_str,
                rew_str,
                eff_str,
                conv_str,
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]JSON report saved to:[/bold green] {report_json}")
        console.print(f"[bold green]CSV report saved to:[/bold green] {report_csv}")

    except Exception as err:
        console.print(f"[bold red]Ablation experiment failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command(name="demo-drone")
def demo_drone(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model artifact (.zip)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to experiment YAML"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for deterministic demo"),
    max_steps: int = typer.Option(150, "--max-steps", help="Maximum steps for demo episode"),
) -> None:
    """Run a deterministic, visual demonstration of the trained drone agent."""
    if not model.exists():
        console.print(f"[bold red]Model file not found:[/bold red] {model}")
        raise typer.Exit(code=1)

    env_kwargs = {}
    if config is not None and config.exists():
        cfg = load_config(config)
        env_kwargs = cfg.environment.parameters

    from adaptive_rl.algorithms.ppo import PPOAlgorithm

    env = make_env("drone", **env_kwargs)
    algo = PPOAlgorithm.from_pretrained(model, env=env)

    obs, info = env.reset(seed=seed)
    console.print(
        Panel.fit(
            f"[bold green]Starting Autonomous Drone 3D Navigation Demo[/bold green]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Seed:[/bold] {seed}\n"
            f"• [bold]Start Position:[/bold] {info.get('position')}\n"
            f"• [bold]Target Waypoint:[/bold] {info.get('goal')}\n"
            f"• [bold]Obstacles in Arena:[/bold] {info.get('num_obstacles')}\n"
            f"• [bold]Initial Distance to Goal:[/bold] {info.get('distance_to_goal', 0):.2f}m",
            title="Demonstration Flight",
            border_style="cyan",
        )
    )

    if hasattr(env, "render"):
        rendered = env.render()
        if rendered:
            console.print(rendered)

    total_reward = 0.0
    outcome = "UNKNOWN"
    step_num = 0

    for step_num in range(1, max_steps + 1):
        action, _ = algo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

        if step_num % 10 == 0 or terminated or truncated:
            dist = step_info.get("distance_to_goal", 0.0)
            alt = step_info.get("altitude", 0.0)
            spd = step_info.get("speed", 0.0)
            min_obs = step_info.get("min_obstacle_distance", 0.0)
            console.print(
                f"Step {step_num:03d} | Alt: {alt:4.1f}m | Spd: {spd:4.1f}m/s | "
                f"Dist: {dist:5.1f}m | Min Obs: {min_obs:4.1f}m | Reward: {reward:+6.2f}"
            )

        if terminated or truncated:
            if step_info.get("success"):
                outcome = "SUCCESS"
            elif step_info.get("collision"):
                outcome = f"FAILED / COLLISION ({step_info.get('collision_type', 'obstacle')})"
            else:
                outcome = "FAILED / MAX STEPS REACHED"
            break

    if hasattr(env, "render"):
        rendered = env.render()
        if rendered:
            console.print(rendered)

    env.close()

    style = "bold green" if outcome == "SUCCESS" else "bold red"
    console.print(
        Panel.fit(
            f"[{style}]OUTCOME: {outcome}[/{style}]\n\n"
            f"• [bold]Total Steps:[/bold] {step_num}\n"
            f"• [bold]Cumulative Reward:[/bold] {total_reward:+.2f}\n"
            f"• [bold]Final Distance to Target:[/bold] {step_info.get('distance_to_goal', 0):.2f}m",
            title="Flight Results",
            border_style="green" if outcome == "SUCCESS" else "red",
        )
    )

    if outcome == "SUCCESS":
        console.print("\n[bold green]SUCCESS[/bold green]")
    else:
        console.print("\n[bold red]FAILED / COLLISION[/bold red]")


@app.command()
def gui(
    port: int = typer.Option(8501, "--port", "-p", help="Port for the Streamlit server."),
    host: str = typer.Option("localhost", "--host", "-h", help="Host address for the server."),
) -> None:
    """Launch the interactive browser-based 3D drone demonstration GUI."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Streamlit is not installed.[/bold red]\n\n"
            "To launch the interactive GUI, install the GUI dependencies:\n"
            '  [bold green]pip install -e ".[gui]"[/bold green]'
        )
        raise typer.Exit(code=1)

    app_path = Path(__file__).resolve().parent.parent.parent / "app.py"
    if not app_path.exists():
        app_path = Path("app.py").resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Launching AdaptiveRL Flight Deck GUI...[/bold cyan]\n\n"
            f"• App Path: {app_path}\n"
            f"• Server URL: http://{host}:{port}\n\n"
            f"[dim]Press Ctrl+C to terminate the server.[/dim]",
            title="Interactive 3D Drone Demonstration",
            border_style="cyan",
        )
    )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]GUI server stopped.[/yellow]")


if __name__ == "__main__":
    app()
