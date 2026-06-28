#!/usr/bin/env python3
"""Run the Aldric agent."""

from __future__ import annotations

from src.agents.aldric import Aldric

def main() -> None:
    agent = Aldric()
    agent.run()


if __name__ == "__main__":
    main()
