"""Public channel-native workflow API with learning and production hardening."""

from harness.natural_channel_service_v2 import *  # noqa: F401,F403
from harness.natural_channel_service_v2 import (
    _explicit_paper_intent,
    _explicit_run_intent,
    _explicit_submit_intent,
    build_natural_channel_service as _build_natural_channel_service,
)


def build_natural_channel_service(config, base_service, *, environ=None):
    service = _build_natural_channel_service(
        config,
        base_service,
        environ=environ,
    )
    from harness.learning_integration import attach_iteration_learning

    service = attach_iteration_learning(
        service,
        config,
        environ=environ,
    )
    from harness.production_hardening import apply_production_hardening

    service = apply_production_hardening(
        service,
        environ=environ,
    )
    from harness.discord_execution_ui import attach_execution_narration_prompt

    return attach_execution_narration_prompt(service)
