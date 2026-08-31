from __future__ import annotations

# FastAPI resolves postponed endpoint annotations from the defining module's
# globals. Export the optional types before calling create_app so importing this
# module works consistently on Python 3.11/3.12.
try:
    from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
except ImportError as exc:  # pragma: no cover - deployment error path
    raise RuntimeError("Install with `pip install -r requirements-platform.txt`") from exc

import harness.platform.api as api_module
from harness.platform.bootstrap import build_service

for _name, _value in {
    "Depends": Depends,
    "FastAPI": FastAPI,
    "Header": Header,
    "HTTPException": HTTPException,
    "WebSocket": WebSocket,
    "WebSocketDisconnect": WebSocketDisconnect,
}.items():
    setattr(api_module, _name, _value)

app = api_module.create_app(build_service())
