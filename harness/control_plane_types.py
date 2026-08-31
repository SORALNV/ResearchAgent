from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = 1


class ControlPlaneError(RuntimeError):
    pass


class NotFoundError(ControlPlaneError):
    pass


class ConflictError(ControlPlaneError):
    pass


class InvalidTransitionError(ControlPlaneError):
    pass


class Domain(str, Enum):
    RESEARCH = "research"
    KAGGLE = "kaggle"
    HYBRID = "hybrid"


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class WorkSessionStatus(str, Enum):
    OPEN = "open"
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    CLOSED = "closed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventLane(str, Enum):
    CONTROL = "control"
    STATUS = "status"
    DATA = "data"
    AUDIT = "audit"


class SteeringKind(str, Enum):
    QUESTION = "question"
    SUPPLEMENT = "supplement"
    CHANGE = "change"
    NEW_HYPOTHESIS = "new_hypothesis"
    CANCEL = "cancel"


class SteeringApplyPolicy(str, Enum):
    READ_ONLY = "read_only"
    IMMEDIATE = "immediate"
    NEXT_CHECKPOINT = "next_checkpoint"
    CHILD_JOB = "child_job"


class SteeringStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


_TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}
_JOB_TRANSITIONS = {
    JobStatus.QUEUED: {
        JobStatus.RUNNING,
        JobStatus.PAUSED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.RUNNING: {
        JobStatus.WAITING_APPROVAL,
        JobStatus.PAUSED,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.WAITING_APPROVAL: {
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.PAUSED: {
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}
_WORK_SESSION_TRANSITIONS = {
    WorkSessionStatus.OPEN: {
        WorkSessionStatus.ACTIVE,
        WorkSessionStatus.PAUSED,
        WorkSessionStatus.CLOSED,
    },
    WorkSessionStatus.ACTIVE: {
        WorkSessionStatus.PAUSED,
        WorkSessionStatus.BLOCKED,
        WorkSessionStatus.CLOSED,
    },
    WorkSessionStatus.PAUSED: {
        WorkSessionStatus.ACTIVE,
        WorkSessionStatus.CLOSED,
    },
    WorkSessionStatus.BLOCKED: {
        WorkSessionStatus.ACTIVE,
        WorkSessionStatus.PAUSED,
        WorkSessionStatus.CLOSED,
    },
    WorkSessionStatus.CLOSED: set(),
}
_STEERING_DEFAULT_POLICY = {
    SteeringKind.QUESTION: SteeringApplyPolicy.READ_ONLY,
    SteeringKind.SUPPLEMENT: SteeringApplyPolicy.NEXT_CHECKPOINT,
    SteeringKind.CHANGE: SteeringApplyPolicy.NEXT_CHECKPOINT,
    SteeringKind.NEW_HYPOTHESIS: SteeringApplyPolicy.CHILD_JOB,
    SteeringKind.CANCEL: SteeringApplyPolicy.IMMEDIATE,
}
