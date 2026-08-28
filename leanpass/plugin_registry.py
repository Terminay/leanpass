from typing import Callable, Any, Dict

# The central registry holding all ops (both built-in and plugins)
_REGISTRY: Dict[str, Callable] = {}

def register_op(name: str, fn: Callable):
    """Registers a function into LeanPass."""
    if name in _REGISTRY:
        pass
    _REGISTRY[name] = fn

def op(name: str):
    """Decorator for registering ops."""
    def decorator(fn):
        register_op(name, fn)
        return fn
    return decorator

def get_op(name: str) -> Callable:
    """Retrieve an op by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Op '{name}' not found. Did you install its plugin?")
    return _REGISTRY[name]

