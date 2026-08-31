from __future__ import annotations

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install with `pip install -r requirements-platform.txt`") from exc

import harness.platform.api as api_module
from harness.platform.api_extensions import register_portable_routes
from harness.platform.application import build_application

for _name, _value in {
    "Depends": Depends,
    "FastAPI": FastAPI,
    "Header": Header,
    "HTTPException": HTTPException,
    "WebSocket": WebSocket,
    "WebSocketDisconnect": WebSocketDisconnect,
}.items():
    setattr(api_module, _name, _value)

service = build_application()
app = api_module.create_app(service)
register_portable_routes(app, service)
