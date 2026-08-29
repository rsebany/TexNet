#!/usr/bin/env python3
"""One-command entry point: ``python main.py`` == ``ildexnet`` console script."""

from ildexnet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())