# resontinex/plugins.py
from importlib.metadata import entry_points

def load_plugins():
    for ep in entry_points(group="resontinex.plugins"):
        ep.load()()