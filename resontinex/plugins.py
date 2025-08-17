"""Cross-version compatible plugin loading system."""

import sys
from typing import Iterator, Optional, Dict, Any

def _iter_plugins() -> Iterator:
    """Iterator for plugins with cross-version compatibility."""
    try:
        # Try the modern importlib.metadata approach first
        from importlib.metadata import entry_points
        
        if sys.version_info >= (3, 10):
            # Python 3.10+ has the select() method
            eps = entry_points()
            if hasattr(eps, "select"):
                yield from eps.select(group="resontinex.plugins")
                return
        
        # Fallback for older versions - try dictionary-style access
        try:
            all_eps = entry_points()
            # Check if it's the old dictionary-style return
            if isinstance(all_eps, dict):
                yield from all_eps.get("resontinex.plugins", [])
            else:
                # Try accessing as attribute (some versions)
                plugins = getattr(all_eps, "resontinex.plugins", [])
                yield from plugins
        except (AttributeError, TypeError):
            pass
            
    except ImportError:
        pass
        
    # Ultimate fallback to pkg_resources
    try:
        import pkg_resources
        for ep in pkg_resources.iter_entry_points("resontinex.plugins"):
            yield ep
    except ImportError:
        pass

def load_plugins(config: Optional[Dict[str, Any]] = None):
    """Load plugins with optional configuration filtering."""
    if config and not config.get("enable_governance", False):
        return
    
    loaded_count = 0
    for ep in _iter_plugins():
        try:
            plugin_name = getattr(ep, 'name', 'unknown')
            print(f"⚙️ Loading plugin: {plugin_name}")
            plugin_init = ep.load()
            if callable(plugin_init):
                plugin_init()
            loaded_count += 1
        except Exception as e:
            plugin_name = getattr(ep, 'name', 'unknown')
            print(f"❌ Failed to load plugin {plugin_name}: {e}")
    
    if loaded_count == 0:
        print("ℹ️ No plugins loaded (governance disabled or no plugins found)")