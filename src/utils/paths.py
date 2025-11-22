# src/utils/paths.py
import os

def get_project_path(*subpath):
    """
    Returns the absolute path inside the project given a relative subpath.
    Usage:
        get_project_path("logs", "rejected_snli.txt")
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return os.path.join(root, *subpath)
