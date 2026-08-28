from .plugin_registry import get_op

def __getattr__(name: str):
    try:
        return get_op(name)
    except KeyError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

