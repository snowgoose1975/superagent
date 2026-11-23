"""Disabled v1 compatibility gateway. The school gateway is in school_gateway.py."""


class DisabledGateway:
    """Compatibility class for the disabled model gateway."""

    def call_llm(self, *args, **kwargs):
        raise NotImplementedError("This compatibility gateway is disabled.")


def call_llm(*args, **kwargs):
    """Compatibility function for the disabled model gateway."""
    raise NotImplementedError("This compatibility gateway is disabled.")
