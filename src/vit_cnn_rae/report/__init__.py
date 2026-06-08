"""Report web app: browse outputs/ runs and visualize metrics / loss / images / logs.

`create_app` is lazily imported so `import vit_cnn_rae.report` (e.g. in tests that
only need `parse`) does not pull in Flask.
"""
from .parse import RunInfo, discover_runs

__all__ = ["RunInfo", "discover_runs", "create_app"]


def __getattr__(name):  # PEP 562
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
