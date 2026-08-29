#!/usr/bin/env python3
"""Convenience one-command entry point for ILD-TexNet.

Equivalent to ``python -m ildexnet.cli``; also exposed as the ``ildexnet``
console script after ``pip install -e .``.
"""

from ildexnet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())