"""Module entry point: python -m theseus.build_deployment BUNDLE."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from theseus.deployment_bundle import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Theseus Compose bundle")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        print(build_bundle(args.bundle))
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        parser.exit(1, f"build-deployment: {detail}\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"build-deployment: {exc}\n")


if __name__ == "__main__":
    main()
