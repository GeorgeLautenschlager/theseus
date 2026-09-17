"""Host-side managed deployment controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from theseus.deployment_control import DeploymentController


def _controller(bundle: Path, root: Path | None) -> DeploymentController:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "deployment.json").read_text(encoding="utf-8"))
    if manifest.get("_generated_by") != "theseus-compose-assembler":
        raise ValueError("bundle is not an assembler-owned deployment")
    deployment_id = manifest["id"]
    deployment_root = root or Path(
        os.environ.get("THESEUS_DEPLOYMENT_ROOT", f"/srv/theseus/{deployment_id}")
    )
    return DeploymentController(
        root=deployment_root,
        deployment_id=deployment_id,
        agent_ids=tuple(manifest["agents"]),
        compose_file=bundle / "compose.yaml",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Control a managed Theseus deployment")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("action", choices=("start", "stop", "status", "retire", "recovery"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--reason")
    parser.add_argument("services", nargs="*")
    args = parser.parse_args()
    try:
        controller = _controller(args.bundle, args.root)
        if args.action == "start":
            value = controller.start(args.services)
        elif args.action == "stop":
            value = controller.stop(timeout_seconds=args.timeout)
        elif args.action == "retire":
            if not args.reason:
                parser.error("retire requires --reason")
            value = controller.retire(reason=args.reason)
        elif args.action == "recovery":
            value = controller.interrupted_operation()
        else:
            value = controller.status()
        if hasattr(value, "__dataclass_fields__"):
            from dataclasses import asdict
            value = asdict(value)
        print(json.dumps(value, indent=2, sort_keys=True))
    except (ValueError, PermissionError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"theseus-deployment: {exc}\n")


if __name__ == "__main__":
    main()
