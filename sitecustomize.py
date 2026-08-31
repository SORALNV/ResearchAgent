"""Portable deployment compatibility hooks.

Python imports sitecustomize automatically when /app is on PYTHONPATH. This
keeps the original command entrypoints working while selecting the portable
remote-worker bundle protocol and generic Discord approval view. Source installs
outside the container behave the same when the repository root is on sys.path.
"""

from __future__ import annotations


def _activate() -> None:
    try:
        import harness.platform.application as application
        from harness.compute.remote_portable import PortableRemoteWorkerBackend

        application.RemoteWorkerBackend = PortableRemoteWorkerBackend
    except Exception:
        pass

    try:
        import harness.compute.worker_api as worker_api
        from harness.compute.worker_api_portable import PortableWorkerJobManager

        worker_api.WorkerJobManager = PortableWorkerJobManager
    except Exception:
        pass

    try:
        from harness.platform.discord_edge import DiscordEdgeBot
        from harness.platform.discord_edge_portable import PortableDiscordEdgeBot

        DiscordEdgeBot._computer_approval_view = (
            PortableDiscordEdgeBot._computer_approval_view
        )
    except Exception:
        pass


_activate()
