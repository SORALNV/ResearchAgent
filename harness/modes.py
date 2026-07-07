from enum import StrEnum


class Mode(StrEnum):
    PLANNING = "PLANNING"
    RESEARCH = "RESEARCH"
    APPROVAL_BLOCKED = "APPROVAL_BLOCKED"
    PAUSED = "PAUSED"
    DONE = "DONE"


ALLOWED_TRANSITIONS: dict[Mode, set[Mode]] = {
    Mode.PLANNING: {Mode.RESEARCH, Mode.APPROVAL_BLOCKED, Mode.PAUSED, Mode.DONE},
    Mode.RESEARCH: {Mode.APPROVAL_BLOCKED, Mode.PAUSED, Mode.DONE, Mode.PLANNING},
    Mode.APPROVAL_BLOCKED: {Mode.RESEARCH, Mode.PLANNING, Mode.DONE},
    Mode.PAUSED: {Mode.PLANNING, Mode.RESEARCH, Mode.APPROVAL_BLOCKED, Mode.DONE},
    Mode.DONE: {Mode.PLANNING},
}


def can_transition(current: Mode, target: Mode) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def require_transition(current: Mode, target: Mode) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid mode transition: {current.value} -> {target.value}")
