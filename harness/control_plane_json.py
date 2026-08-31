from __future__ import annotations

import json
from typing import Any, Mapping


def json_dict(value: Mapping[str, Any] | object | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): json_copy(item) for key, item in value.items()}
    return {}


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
