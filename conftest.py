"""Pytest configuration: make the ``cffdrs`` package importable from ``src/``.

An editable install (``uv sync``) also makes ``cffdrs`` importable; this fallback
lets ``python -m pytest`` work on a bare checkout without an install step.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
