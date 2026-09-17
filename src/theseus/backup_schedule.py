"""Render and install an opt-in systemd timer for scheduled deployment backups.

Generated unit files never contain credential values: object-store credentials
stay in an operator-referenced ``EnvironmentFile`` path (never its contents) or
the process environment the timer's service inherits. A fresh install never
enables or starts the timer; the operator must pass ``enable=True`` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
from typing import Callable, Sequence


DEFAULT_ON_CALENDAR = "daily"
DEFAULT_EXECUTABLE = "/opt/theseus-operator/bin/theseus-backup-schedule"


def _unit_names(deployment_id: str) -> tuple[str, str]:
    return f"theseus-backup-{deployment_id}.service", f"theseus-backup-{deployment_id}.timer"


def _scope_flag(scope: str) -> tuple[str, ...]:
    if scope not in ("system", "user"):
        raise ValueError("scope must be 'system' or 'user'")
    return ("--user",) if scope == "user" else ()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_service_unit(
    deployment_id: str,
    bundle: Path,
    root: Path,
    *,
    executable: str = DEFAULT_EXECUTABLE,
    env_file: Path | None = None,
    store_args: Sequence[str] = (),
) -> str:
    """One-shot unit that attempts exactly one eligibility-checked backup."""
    command = [executable, str(bundle), "run", "--root", str(root), *store_args]
    lines = [
        "[Unit]",
        f"Description=Theseus scheduled backup for {deployment_id}",
        "",
        "[Service]",
        "Type=oneshot",
    ]
    if env_file is not None:
        lines.append(f"EnvironmentFile=-{env_file}")
    lines.append(f"ExecStart={' '.join(shlex.quote(part) for part in command)}")
    return "\n".join(lines) + "\n"


def render_timer_unit(
    deployment_id: str,
    *,
    on_calendar: str = DEFAULT_ON_CALENDAR,
    persistent: bool = True,
    randomized_delay_sec: int = 0,
) -> str:
    if not on_calendar or not on_calendar.strip() or "\n" in on_calendar:
        raise ValueError("on_calendar must be a nonempty single-line systemd calendar expression")
    if randomized_delay_sec < 0:
        raise ValueError("randomized_delay_sec must not be negative")
    service_name, _ = _unit_names(deployment_id)
    lines = [
        "[Unit]",
        f"Description=Theseus scheduled backup timer for {deployment_id}",
        "",
        "[Timer]",
        f"Unit={service_name}",
        f"OnCalendar={on_calendar}",
        f"Persistent={'true' if persistent else 'false'}",
    ]
    if randomized_delay_sec:
        lines.append(f"RandomizedDelaySec={randomized_delay_sec}")
    lines += ["", "[Install]", "WantedBy=timers.target"]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TimerInstallation:
    unit_dir: Path
    service_path: Path
    timer_path: Path
    service_name: str
    timer_name: str
    enabled: bool


def install(
    deployment_id: str,
    bundle: Path,
    root: Path,
    unit_dir: Path,
    *,
    on_calendar: str = DEFAULT_ON_CALENDAR,
    executable: str = DEFAULT_EXECUTABLE,
    env_file: Path | None = None,
    store_args: Sequence[str] = (),
    scope: str = "system",
    enable: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> TimerInstallation:
    """Write the timer/service unit pair. Leaves the timer disabled unless asked.

    Reuses the deployment's own operation lock and the normal backup protocol:
    the generated service only ever invokes ``theseus-backup-schedule ... run``,
    which is the same eligibility-checked path exercised by ``scheduled_backup``.
    """
    flag = _scope_flag(scope)
    unit_dir = Path(unit_dir)
    service_name, timer_name = _unit_names(deployment_id)
    service_path = unit_dir / service_name
    timer_path = unit_dir / timer_name
    _atomic_text(
        service_path,
        render_service_unit(
            deployment_id, bundle, root,
            executable=executable, env_file=env_file, store_args=store_args,
        ),
    )
    _atomic_text(timer_path, render_timer_unit(deployment_id, on_calendar=on_calendar))
    run(["systemctl", *flag, "daemon-reload"], check=True, text=True, capture_output=True)
    if enable:
        run(["systemctl", *flag, "enable", "--now", timer_name], check=True, text=True, capture_output=True)
    return TimerInstallation(unit_dir, service_path, timer_path, service_name, timer_name, enabled=enable)


def uninstall(
    deployment_id: str,
    unit_dir: Path,
    *,
    scope: str = "system",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    flag = _scope_flag(scope)
    unit_dir = Path(unit_dir)
    service_name, timer_name = _unit_names(deployment_id)
    run(["systemctl", *flag, "disable", "--now", timer_name], check=False, text=True, capture_output=True)
    (unit_dir / timer_name).unlink(missing_ok=True)
    (unit_dir / service_name).unlink(missing_ok=True)
    run(["systemctl", *flag, "daemon-reload"], check=True, text=True, capture_output=True)


def status(
    deployment_id: str,
    unit_dir: Path,
    *,
    scope: str = "system",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    flag = _scope_flag(scope)
    unit_dir = Path(unit_dir)
    service_name, timer_name = _unit_names(deployment_id)

    def query(unit: str, field: str) -> str | None:
        try:
            result = run(
                ["systemctl", *flag, "show", unit, f"--property={field}", "--value"],
                check=True, text=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        value = result.stdout.strip()
        return value or None

    return {
        "deployment_id": deployment_id,
        "service_name": service_name,
        "timer_name": timer_name,
        "installed": (unit_dir / service_name).is_file() and (unit_dir / timer_name).is_file(),
        "enabled": query(timer_name, "UnitFileState"),
        "active": query(timer_name, "ActiveState"),
        "next_run": query(timer_name, "NextElapseUSecRealtime"),
        "last_result": query(service_name, "Result"),
    }
