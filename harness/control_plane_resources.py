from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class ResourceRequirements:
    cpu_cores: float | None = None
    memory_mb: int | None = None
    gpu_count: int = 0
    gpu_memory_mb: int | None = None
    accelerator: str | None = None
    ephemeral_storage_mb: int | None = None
    network_required: bool = False
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        for name in ("memory_mb", "gpu_memory_mb", "ephemeral_storage_mb"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "accelerator": self.accelerator,
            "ephemeral_storage_mb": self.ephemeral_storage_mb,
            "network_required": self.network_required,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ResourceRequirements":
        value = dict(data or {})
        return cls(
            cpu_cores=(float(value["cpu_cores"]) if value.get("cpu_cores") is not None else None),
            memory_mb=(int(value["memory_mb"]) if value.get("memory_mb") is not None else None),
            gpu_count=int(value.get("gpu_count") or 0),
            gpu_memory_mb=(
                int(value["gpu_memory_mb"])
                if value.get("gpu_memory_mb") is not None
                else None
            ),
            accelerator=(str(value["accelerator"]) if value.get("accelerator") else None),
            ephemeral_storage_mb=(
                int(value["ephemeral_storage_mb"])
                if value.get("ephemeral_storage_mb") is not None
                else None
            ),
            network_required=bool(value.get("network_required", False)),
            labels=tuple(str(item) for item in value.get("labels", [])),
        )
