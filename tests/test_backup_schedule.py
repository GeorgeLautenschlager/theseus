from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from theseus.backup_schedule import (
    install,
    render_service_unit,
    render_timer_unit,
    status,
    uninstall,
)


class FakeSystemctl:
    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        stdout = ""
        if "show" in command:
            field = next(part for part in command if part.startswith("--property="))
            stdout = {
                "--property=UnitFileState": "disabled",
                "--property=ActiveState": "inactive",
                "--property=NextElapseUSecRealtime": "",
                "--property=Result": "success",
            }.get(field, "")
        return subprocess.CompletedProcess(command, 0, stdout, "")


def test_service_unit_never_embeds_credential_values():
    unit = render_service_unit(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        env_file=Path("/etc/theseus/flywheel-backup.env"),
        store_args=("--endpoint", "https://example.r2.cloudflarestorage.com", "--bucket", "backups"),
    )
    assert "ExecStart=" in unit
    assert "build/flywheel" in unit
    assert "/srv/theseus/flywheel" in unit
    assert "EnvironmentFile=-/etc/theseus/flywheel-backup.env" in unit
    for secret_like in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"):
        assert secret_like not in unit


def test_timer_unit_renders_calendar_and_references_service():
    unit = render_timer_unit("flywheel", on_calendar="03:00")
    assert "OnCalendar=03:00" in unit
    assert "Unit=theseus-backup-flywheel.service" in unit
    assert "Persistent=true" in unit
    assert "WantedBy=timers.target" in unit


def test_timer_unit_rejects_empty_or_multiline_calendar():
    with pytest.raises(ValueError):
        render_timer_unit("flywheel", on_calendar="")
    with pytest.raises(ValueError):
        render_timer_unit("flywheel", on_calendar="daily\nweekly")


def test_install_writes_units_and_is_disabled_by_default(tmp_path):
    systemctl = FakeSystemctl()
    installed = install(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        tmp_path / "units", run=systemctl,
    )
    assert installed.enabled is False
    assert installed.service_path.is_file()
    assert installed.timer_path.is_file()
    assert ["systemctl", "daemon-reload"] == systemctl.commands[0]
    assert all("enable" not in command for command in systemctl.commands)


def test_install_can_enable_and_start_when_asked(tmp_path):
    systemctl = FakeSystemctl()
    installed = install(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        tmp_path / "units", run=systemctl, enable=True,
    )
    assert installed.enabled is True
    assert any(
        "enable" in command and "--now" in command and "theseus-backup-flywheel.timer" in command
        for command in systemctl.commands
    )


def test_install_user_scope_passes_user_flag(tmp_path):
    systemctl = FakeSystemctl()
    install(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        tmp_path / "units", run=systemctl, scope="user", enable=True,
    )
    assert all("--user" in command for command in systemctl.commands)


def test_uninstall_removes_unit_files_and_disables(tmp_path):
    systemctl = FakeSystemctl()
    installed = install(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        tmp_path / "units", run=systemctl, enable=True,
    )
    uninstall("flywheel", tmp_path / "units", run=systemctl)
    assert not installed.service_path.exists()
    assert not installed.timer_path.exists()
    assert any("disable" in command for command in systemctl.commands)


def test_status_reports_installed_files_and_queried_unit_state(tmp_path):
    systemctl = FakeSystemctl()
    install(
        "flywheel", Path("build/flywheel"), Path("/srv/theseus/flywheel"),
        tmp_path / "units", run=systemctl,
    )
    value = status("flywheel", tmp_path / "units", run=systemctl)
    assert value["installed"] is True
    assert value["enabled"] == "disabled"
    assert value["active"] == "inactive"


def test_status_when_never_installed(tmp_path):
    systemctl = FakeSystemctl()
    value = status("flywheel", tmp_path / "units", run=systemctl)
    assert value["installed"] is False
