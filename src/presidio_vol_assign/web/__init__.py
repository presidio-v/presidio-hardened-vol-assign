"""Optional web GUI for the demo server (``pva serve``).

Install with the ``web`` extra::

    pip install "presidio-hardened-vol-assign[web]"

Nothing in this package is imported by the core library or the solver CLI, so
the extra stays genuinely optional.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    # Lazy re-export so importing this package does not require FastAPI unless
    # the caller actually asks for the app factory.
    if name == "create_app":
        from presidio_vol_assign.web.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
