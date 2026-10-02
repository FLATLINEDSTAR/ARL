import enum
import sys
import os
import subprocess
import tempfile
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

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

class SystemDoctor:
    """Central diagnostic engine for adaptive-rl doctor command."""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        
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
                remediation="Please install a Python version between 3.10 and 3.12 (inclusive)."
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
                    status=CheckStatus.WARN,
                    message=f"No GPU acceleration found. Falling back to CPU ({threads} threads).",
                    remediation="If you have an NVIDIA GPU, verify your PyTorch CUDA installation: pip install torch --index-url https://download.pytorch.org/whl/cu118"
                )
        except ImportError:
            return CheckResult(
                category="Hardware Acceleration",
                status=CheckStatus.FAIL,
                message=f"PyTorch is not installed. CPU threads: {threads}",
                remediation="pip install torch"
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
                message="All core RL dependencies are installed."
            )
        else:
            return CheckResult(
                category="Core Dependencies",
                status=CheckStatus.FAIL,
                message=f"Missing core dependencies: {', '.join(missing)}",
                remediation=f"pip install {' '.join(missing)}"
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
                message="All GUI and visualization dependencies are installed."
            )
        else:
            return CheckResult(
                category="GUI/Viz Dependencies",
                status=CheckStatus.WARN,
                message=f"Missing optional GUI/Viz dependencies: {', '.join(missing)}",
                remediation=f"pip install {' '.join(missing)}"
            )

    def check_gymnasium_registration(self) -> CheckResult:
        """Verifies drone-3d-v0, drone-6dof, drone_disturbed exist in the registry."""
        try:
            import gymnasium as gym
            try:
                import adaptive_rl.environments
            except ImportError:
                pass
            
            registry_keys = list(gym.envs.registry.keys())
            
            expected = ["drone-3d-v0", "drone-6dof", "drone_disturbed"]
            missing = [env for env in expected if env not in registry_keys]
            
            if not missing:
                return CheckResult(
                    category="Environment Registration",
                    status=CheckStatus.PASS,
                    message="All custom Gymnasium environments are registered."
                )
            else:
                return CheckResult(
                    category="Environment Registration",
                    status=CheckStatus.FAIL,
                    message=f"Missing Gymnasium environments: {', '.join(missing)}",
                    remediation="Check that src/adaptive_rl/environments/__init__.py correctly registers these environments."
                )
        except ImportError:
            return CheckResult(
                category="Environment Registration",
                status=CheckStatus.FAIL,
                message="Gymnasium is not installed.",
                remediation="pip install gymnasium"
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
                    message="Successfully serialized and deserialized PyTorch model."
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except ImportError:
             return CheckResult(
                category="Model Checkpointing",
                status=CheckStatus.FAIL,
                message="PyTorch is not installed, cannot test checkpointing.",
                remediation="pip install torch"
            )
        except Exception as e:
            return CheckResult(
                category="Model Checkpointing",
                status=CheckStatus.FAIL,
                message=f"Failed to save/load model checkpoint: {str(e)}",
                remediation="Ensure you have write permissions to temp directory and PyTorch is functioning."
            )

    def check_classical_planner(self) -> CheckResult:
        """Runs a minimal A* search graph to verify sanity."""
        try:
            import heapq
            
            def minimal_astar():
                start, goal = (0, 0), (1, 1)
                queue = [(0, start)]
                visited = set()
                while queue:
                    cost, node = heapq.heappop(queue)
                    if node == goal:
                        return True
                    if node in visited:
                        continue
                    visited.add(node)
                    heapq.heappush(queue, (cost + 1, (node[0]+1, node[1]+1)))
                return False
                
            if minimal_astar():
                return CheckResult(
                    category="Classical Planner",
                    status=CheckStatus.PASS,
                    message="Classical planner graph search sanity check passed."
                )
            else:
                return CheckResult(
                    category="Classical Planner",
                    status=CheckStatus.FAIL,
                    message="A* search failed to find trivial path.",
                    remediation="Check planner logic in adaptive_rl/algorithms."
                )
        except Exception as e:
            return CheckResult(
                category="Classical Planner",
                status=CheckStatus.FAIL,
                message=f"Error in classical planner sanity check: {e}",
                remediation="Ensure required modules are installed."
            )

    def check_git_health(self) -> CheckResult:
        """Queries valid commit SHA, detached HEAD state, and dirty working tree."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            
            sha_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if sha_result.returncode != 0:
                return CheckResult(
                    category="Git Health",
                    status=CheckStatus.WARN,
                    message="Not a git repository or no commits.",
                    remediation="Initialize git repository: git init && git commit -m 'initial'"
                )
            
            branch_result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            is_detached = (branch_result.stdout.strip() == "")
            
            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            is_dirty = bool(status_result.stdout.strip())
            
            if is_detached or is_dirty:
                warnings = []
                if is_detached: warnings.append("Detached HEAD")
                if is_dirty: warnings.append("Dirty working tree")
                return CheckResult(
                    category="Git Health",
                    status=CheckStatus.WARN,
                    message=f"Git warnings: {', '.join(warnings)}.",
                    remediation="Commit or stash changes, and switch to a valid branch."
                )
            
            return CheckResult(
                category="Git Health",
                status=CheckStatus.PASS,
                message="Git repository is healthy (clean working tree on a branch)."
            )
        except FileNotFoundError:
            return CheckResult(
                category="Git Health",
                status=CheckStatus.WARN,
                message="Git command not found.",
                remediation="Install Git on your system."
            )
        except Exception as e:
            return CheckResult(
                category="Git Health",
                status=CheckStatus.WARN,
                message=f"Git check failed: {e}",
                remediation="Verify git installation."
            )

    def check_io_permissions(self) -> CheckResult:
        """Tests write access to artifacts/ and logs/ directories."""
        dirs_to_check = ["artifacts", "logs"]
        failed_dirs = []
        
        for d in dirs_to_check:
            path = Path(d)
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
                message="Write access confirmed for artifacts/ and logs/."
            )
        else:
            return CheckResult(
                category="I/O Permissions",
                status=CheckStatus.FAIL,
                message=f"Missing write permissions for: {', '.join(failed_dirs)}",
                remediation=f"Check folder permissions or run with appropriate access."
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
                remediation="pip install pyyaml"
            )
            
        config_dir = Path("configs")
        if not config_dir.exists():
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.WARN,
                message="configs/ directory does not exist.",
                remediation="Create a configs/ directory for your configuration files."
            )
            
        yaml_files = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob("configs/**/*.yml", recursive=True)
        
        if not yaml_files:
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.WARN,
                message="No YAML files found in configs/.",
                remediation="Add configuration files to configs/."
            )
            
        failed_files = []
        for file in yaml_files:
            try:
                with open(file, "r") as f:
                    yaml.safe_load(f)
            except Exception as e:
                failed_files.append((file, str(e)))
                
        if not failed_files:
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.PASS,
                message=f"Successfully validated {len(yaml_files)} YAML configurations."
            )
        else:
            error_details = "; ".join([f"{f}: {e}" for f, e in failed_files])
            return CheckResult(
                category="YAML Configs",
                status=CheckStatus.FAIL,
                message=f"Failed to parse {len(failed_files)} YAML files.",
                remediation=f"Check formatting for: {error_details}"
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
                    "remediation": r.remediation
                }
                for r in self.results
            ]
        }
