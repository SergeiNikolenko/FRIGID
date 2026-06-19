"""FRIGID public package entrypoints."""

__all__ = ["__version__"]

try:
    from importlib.metadata import version

    __version__ = version("frigid")
except Exception:
    __version__ = "0+unknown"
