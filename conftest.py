"""
conftest.py — adds project root to sys.path so tests can import the enhancer package
without requiring pip install.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
