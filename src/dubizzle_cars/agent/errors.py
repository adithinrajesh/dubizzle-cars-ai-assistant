"""Safe agent errors: messages never contain model arguments or SDK exception text."""


class AgentError(Exception):
    """The assistant could not safely complete this request."""


class AgentProviderError(AgentError):
    """Missing configuration, provider failure, or malformed model response."""


class ToolValidationError(AgentError):
    """Unknown tool or invalid arguments."""


class ToolExecutionError(AgentError):
    """An inventory tool could not complete safely."""


class GroundingError(AgentError):
    """The final response references unsupported inventory data."""


class ToolRoundLimitError(AgentError):
    """The bounded tool loop exhausted its budget."""
