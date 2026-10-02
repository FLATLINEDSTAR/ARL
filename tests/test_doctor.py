import json
from unittest.mock import MagicMock, patch

import pytest

from adaptive_rl.diagnostics.doctor import CheckStatus, SystemDoctor


@pytest.fixture
def doctor():
    return SystemDoctor()


def test_check_python_version_pass(doctor):
    with patch("sys.version_info") as mock_version:
        mock_version.major = 3
        mock_version.minor = 10
        mock_version.micro = 0
        result = doctor.check_python_version()
        assert result.status == CheckStatus.PASS


def test_check_python_version_fail(doctor):
    with patch("sys.version_info") as mock_version:
        mock_version.major = 3
        mock_version.minor = 9
        mock_version.micro = 0
        result = doctor.check_python_version()
        assert result.status == CheckStatus.FAIL


def test_check_hardware_acceleration_cuda(doctor):
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        import torch

        torch.cuda.is_available.return_value = True
        result = doctor.check_hardware_acceleration()
        assert result.status == CheckStatus.PASS
        assert "CUDA" in result.message


def test_check_hardware_acceleration_cpu_only(doctor):
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        import torch

        torch.cuda.is_available.return_value = False
        torch.backends = MagicMock()
        torch.backends.mps.is_available.return_value = False
        result = doctor.check_hardware_acceleration()
        assert result.status == CheckStatus.WARN
        assert "CPU" in result.message


def test_check_core_dependencies_pass(doctor):
    with patch("builtins.__import__", return_value=MagicMock()):
        result = doctor.check_core_dependencies()
        assert result.status == CheckStatus.PASS


def test_check_core_dependencies_fail(doctor):
    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named torch")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mock_import):
        result = doctor.check_core_dependencies()
        assert result.status == CheckStatus.FAIL
        assert "torch" in result.message


def test_check_gui_dependencies_warn(doctor):
    def mock_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("No module named streamlit")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mock_import):
        result = doctor.check_gui_dependencies()
        assert result.status == CheckStatus.WARN
        assert "streamlit" in result.message


def test_check_gymnasium_registration_fail(doctor):
    with patch.dict("sys.modules", {"gymnasium": MagicMock()}):
        import gymnasium as gym

        gym.envs.registry.keys.return_value = []
        result = doctor.check_gymnasium_registration()
        assert result.status == CheckStatus.FAIL
        assert "drone-3d-v0" in result.message


def test_check_model_checkpointing_fail(doctor):
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        import torch

        torch.save.side_effect = Exception("Disk full")
        result = doctor.check_model_checkpointing()
        assert result.status == CheckStatus.FAIL
        assert "Disk full" in result.message


def test_check_classical_planner_pass(doctor):
    result = doctor.check_classical_planner()
    assert result.status == CheckStatus.PASS


def test_check_io_permissions_pass(doctor, tmp_path):
    with patch("adaptive_rl.diagnostics.doctor.Path", return_value=tmp_path):
        result = doctor.check_io_permissions()
        assert result.status == CheckStatus.PASS


def test_check_yaml_configs_warn(doctor):
    with patch("glob.glob", return_value=[]):
        result = doctor.check_yaml_configs()
        assert result.status == CheckStatus.WARN


def test_json_export_and_dict(doctor):
    with patch("sys.version_info") as mock_version:
        mock_version.major = 3
        mock_version.minor = 10
        doctor.add_result(doctor.check_python_version())

    summary = doctor.to_dict()
    assert "total" in summary
    assert "passed" in summary
    assert "warnings" in summary
    assert "failed" in summary
    assert "results" in summary
    assert summary["total"] == 1

    json_str = json.dumps(summary)
    assert "Python Version" in json_str


def test_cli_exit_code_pass():
    from typer.testing import CliRunner

    from adaptive_rl.cli import app

    runner = CliRunner()

    with patch("adaptive_rl.diagnostics.doctor.SystemDoctor.run_all_checks") as mock_run:
        from adaptive_rl.diagnostics.doctor import CheckResult, CheckStatus

        mock_run.return_value = [
            CheckResult(category="Test", status=CheckStatus.PASS, message="OK")
        ]
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0


def test_cli_exit_code_fail():
    from typer.testing import CliRunner

    from adaptive_rl.cli import app

    runner = CliRunner()

    with patch("adaptive_rl.diagnostics.doctor.SystemDoctor.run_all_checks") as mock_run:
        from adaptive_rl.diagnostics.doctor import CheckResult, CheckStatus

        mock_run.return_value = [
            CheckResult(category="Test", status=CheckStatus.FAIL, message="Fail")
        ]
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
