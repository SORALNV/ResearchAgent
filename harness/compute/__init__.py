"""Portable compute backends for Kaggle, remote GPUs, GPU VMs, and local CPU."""

from harness.compute.base import (
    BackendCapabilities,
    BackendStatus,
    ComputeBackend,
    ComputeBroker,
    ComputeHandle,
)
from harness.compute.fake import FakeComputeBackend
from harness.compute.scheduler import JobScheduler

__all__ = [
    "BackendCapabilities",
    "BackendStatus",
    "ComputeBackend",
    "ComputeBroker",
    "ComputeHandle",
    "FakeComputeBackend",
    "JobScheduler",
]
