import enum
import glob
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


class CheckStatus(str, enum.Enum):
    """Status of a diagnostic check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """Standardized data structure for a diagnostic check result."""

    category: str
    status: CheckStatus
    message: str
    remediation: str = ""


def get_project_root() -> Path:
    """Determine the root directory of the ARL project."""
    # Assuming this file is at src/adaptive_rl/diagnostics/doctor.py
    return Path(__file__).resolve().parent.parent.parent.parent


class SystemDoctor:
    """Central diagnostic engine for adaptive-rl doctor command."""

    def __init__(self) -> None:
        self.results: List[CheckResult] = []
        self.project_root = get_project_root()

    def run_all_checks(self) -> List[CheckResult]:
        """Runs all registered diagnostic checks."""
        self.results.clear()

        self.add_result(self.check_python_version())
        self.add_result(self.check_hardware_acceleration())
        self.add_result(self.check_core_dependencies())
        self.add_result(self.check_gui_dependencies())
        self.add_result(self.check_gymnasium_registration())
        self.add_result(self.check_model_checkpointing())
        self.add_result(self.check_classical_planner())
        self.add_result(self.check_git_health())
        self.add_result(self.check_io_permissions())
        self.add_result(self.check_yaml_configs())

        return self.results

    def check_python_version(self) -> CheckResult:
        """Validates Python version is >=3.10 and <=3.12."""
        version = sys.version_info
        if 3 <= version.major <= 3 and 10 <= version.minor <= 12:
            return CheckResult(
                category="Python Version",
                status=CheckStatus.PASS,
                message=f"Python version {version.major}.{version.minor}.{version.micro} is supported.",
            )
        else:
            return CheckResult(
                category="Python Version",
                status=CheckStatus.FAIL,
                message=f"Python version {version.major}.{version.minor}.{version.micro} is unsupported.",
                remediation="Please install a Python version between 3.10 and 3.12 (inclusive).",
            )

    def check_hardware_acceleration(self) -> CheckResult:
        """Detects CUDA availability, Apple MPS, and CPU thread count."""
        threads = os.cpu_count() or 1
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

            accel = "CUDA" if cuda_available else ("MPS" if mps_available else "CPU only")

            if cuda_available or mps_available:
                return CheckResult(
                    category="Hardware Acceleration",
                    status=CheckStatus.PASS,
                    message=f"Hardware acceleration available ({accel}). CPU threads: {threads}",
                )
            else:
                return CheckResult(
                    category="Hardware Acceleration",
                    status=CheckStatus.PASS,
                    message=f"No GPU acceleration found. Operating on CPU ({threads} threads).",
                    remediation="If GPU acceleration is required for your workload, verify your PyTorch installation.",
                )
        except ImportError:
            return CheckResult(
                category="Hardware Acceleration",
                status=CheckStatus.FAIL,
                message=f"PyTorch is not installed. CPU threads: {threads}",
                remediation="Please install the project dependencies (e.g., pip install -e .).",
            )

    def check_core_dependencies(self) -> CheckResult:
        """Tests imports for torch, stable_baselines3, gymnasium, numpy, scipy."""
        deps = ["torch", "stable_baselines3", "gymnasium", "numpy", "scipy"]
        missing = []
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if not missing:
            return CheckResult(
                category="Core Dependencies",
                status=CheckStatus.PASS,
                message="All core RL dependencies are installed.",
            )
        else:
            return CheckResult(
                category="Core Dependencies",
                status=CheckStatus.FAIL,
                message=f"Missing core dependencies: {', '.join(missing)}",
                remediation="Please install the project dependencies (e.g., pip install -e .).",
            )

    def check_gui_dependencies(self) -> CheckResult:
        """Tests imports for streamlit, plotly, rich, typer."""
        deps = ["streamlit", "plotly", "rich", "typer"]
        missing = []
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if not missing:
            return CheckResult(
                category="GUI/Viz Dependencies",
                status=CheckStatus.PASS,
                message="All GUI and visualization dependencies are installed.",
            )
        else:
            return CheckResult(
                category="GUI/Viz Dependencies",
                status=CheckStatus.WARN,
                message=f"Missing optional GUI/Viz dependencies: {', '.join(missing)}",
                remediation="Please install the project's optional GUI dependencies.",
            )

    def check_gymnasium_registration(self) -> CheckResult:
        """Verifies drone-3d-v0, drone-6dof, drone_disturbed exist in the registry."""
        try:
            import importlib.util
            import gymnasium as gym

            if importlib.util.find_spec("adaptive_rl.environments") is not None:
                # Deliberately surface any ImportError in the environments module
                __import__("adaptive_rl.environments")

            registry_keys = list(gym.envs.registry.keys())

            expected = ["drone-3d-v0", "drone-6dof", "drone_disturbed"]
            missing = [env for env in expected if env not in registry_keys]

            if not missing:
                return CheckResult(
                    category="Environment Registration",
                    status=CheckStatus.PASS,
                    message="All custom Gymnasium environments are registered.",
                )
            else:
                return CheckResult(
                    category="Environment Registration",
                    status=CheckStatus.FAIL,
                    message=f"Missing Gymnasium environments: {', '.join(missing)}",
                    remediation="Check that src/adaptive_rl/environments/__init__.py correctly registers these environments.",
                )
        except ImportError as e:
            return CheckResult(
                category="Environment Registration",
                status=CheckStatus.FAIL,
                message=f"Environment registration failed due to import error: {e}",
                remediation="Fix the import error in adaptive_rl.environments or install project dependencies.",
            )

    def check_model_checkpointing(self) -> CheckResult:
        """Serializes and deserializes a tiny dummy PyTorch policy."""
        try:
            import torch
            import torch.nn as nn

            model = nn.Linear(1, 1)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                temp_path = f.name

            try:
                torch.save(model.state_dict(), temp_path)
                # Setting weights_only=True for safety
                loaded_state = torch.load(temp_path, weights_only=True)
                model.load_state_dict(loaded_state)
                return CheckResult(
                    category="Model Checkpointing",
                    status=CheckStatus.PASS,
                    message="Successfully serialized and deserialized PyTorch model.",
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except ImportError:
            return CheckResult(
                category="Model Checkpointing",
                status=CheckStatus.FAIL,
                message="PyTorch is not installed, cannot test checkpointing.",
                remediation="Please install the project dependencies.",
            )
        except Exception as e:
            return CheckResult(
                category="Model Checkpointing",
                status=CheckStatus.FAIL,
                message=f"Failed to save/load model checkpoint: {str(e)}",
                remediation="Ensure you have write permissions to temp directory and PyTorch is functioning.",
            )

    def check_classical_planner(self) -> CheckResult:
        """Tests the actual ARL classical planner with a deterministic fixture."""
        try:
            from adaptive_rl.algorithms.planning import AStarPlanner
            
            planner = AStarPlanner()
            # minimal fixture: 2D grid, start (0,0), goal (1,1)
            path = planner.plan((0,0), (1,1), set())
            if path and path[-1] == (1,1):
                return CheckResult(
                    category="Classical Planner",
                    status=CheckStatus.PASS,
                    message="Classical planner graph search sanity check passed.",
                )
            else:
                return CheckResult(
                    category="Classical Planner",
                    status=CheckStatus.FAIL,
                    message="Planner failed to find trivial path.",
                    remediation="Check planner logic in adaptive_rl/algorithms/planning.py.",
                )
        except ImportError as e:
            return CheckResult(
                category="Classical Planner",
                status=CheckStatus.FAIL,
                message=f"Could not import ARL classical planner: {e}",
                remediation="Ensure adaptive_rl.algorithms.planning is implemented and accessible.",
            )
        except Exception as e:
            return CheckResult(
                category="Classical Planner",
                status=CheckStatus.FAIL,
                message=f"Error in classical planner sanity check: {e}",
                remediation="Check planner logic for errors.",
            )

    def check_git_health(self) -> CheckResult:
        """Queries valid commit SHA, detached HEAD state, and dirty working tree."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True, cwd=self.project_root)

            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=self.project_root
            )
            if sha_result.returncode != 0:
                return CheckResult(
                    category="Git Health",
                    status=CheckStatus.WARN,
                    message="Not a git repository or no commits.",
                    remediation="Initialize git repository: git init && git commit -m 'initial'",
                )

            branch_result = subprocess.run(
                ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=self.project_root
            )
            is_detached = branch_result.stdout.strip() == ""

            status_result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=self.project_root
            )
            is_dirty = bool(status_result.stdout.strip())

            if is_detached or is_dirty:
                warnings = []
                if is_detached:
                    warnings.append("Detached HEAD")
                if is_dirty:
                    warnings.append("Dirty working tree")
                return CheckResult(
                    category="Git Health",
                    status=CheckStatus.WARN,
                    message=f"Git warnings: {', '.join(warnings)}.",
                    remediation="Commit or stash changes, and switch to a valid branch.",
                )

            return CheckResult(
                category="Git Health",
                status=CheckStatus.PASS,
                message="Git repository is healthy (clean working tree on a branch).",
            )
        except FileNotFoundError:
            return CheckResult(
                category="Git Health",
                status=CheckStatus.WARN,
                message="Git command not found.",
                remediation="Install Git on your system.",
            )
        except Exception as e:
            return CheckResult(
                category="Git Health",
                status=CheckStatus.WARN,
                message=f"Git check failed: {e}",
                remediation="Verify git installation.",
            )

    def check_io_permissions(self) -> CheckResult:
        """Tests write access to artifacts/ and logs/ directories."""
        dirs_to_check = ["artifacts", "logs"]
        failed_dirs = []

        for d in dirs_to_check:
            path = self.project_root / d
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except Exception:
                failed_dirs.append(d)

        if not failed_dirs:
            return CheckResult(
                category="I/O Permissions",
                status=CheckStatus.PASS,
                message="Write access confirmed for artifacts/ and logs/.",
            )
        else:
            return CheckResult(
                category="I/O Permissions",
                status=CheckStatus.FAIL,
                message=f"Missing write permissions for: {', '.join(failed_dirs)} in {self.project_root}",
                remediation="Check folder permissions or run with appropriate access.",
            )

    def check_yaml_configs(self) -> CheckResult:
        """Parses all YAML files in configs/ to ensure structural integrity."""
        try:
            import yaml
        except ImportError:
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.FAIL,
                message="PyYAML is not installed.",
                remediation="Please install the project dependencies.",
            )

        config_dir = self.project_root / "configs"
        if not config_dir.exists():
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.WARN,
                message="configs/ directory does not exist.",
                remediation="Create a configs/ directory for your configuration files.",
            )

        yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))

        if not yaml_files:
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.WARN,
                message="No YAML files found in configs/.",
                remediation="Add configuration files to configs/.",
            )

        failed_files = []
        for file in yaml_files:
            try:
                with open(file, "r") as f:
                    yaml.safe_load(f)
            except Exception as e:
                failed_files.append((str(file), str(e)))

        if not failed_files:
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.PASS,
                message=f"Successfully validated {len(yaml_files)} YAML configurations.",
            )
        else:
            error_details = "; ".join([f"{Path(f).name}: {e}" for f, e in failed_files])
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.FAIL,
                message=f"Failed to parse {len(failed_files)} YAML files.",
                remediation=f"Check formatting for: {error_details}",
            )

    def add_result(self, result: CheckResult) -> None:
        """Adds a diagnostic result to the internal list."""
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the diagnostic results to a dictionary structure."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == CheckStatus.PASS),
            "warnings": sum(1 for r in self.results if r.status == CheckStatus.WARN),
            "failed": sum(1 for r in self.results if r.status == CheckStatus.FAIL),
            "results": [
                {
                    "category": r.category,
                    "status": r.status.value,
                    "message": r.message,
                    "remediation": r.remediation,
                }
                for r in self.results
            ],
        }
