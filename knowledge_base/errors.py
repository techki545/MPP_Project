from typing import Any


class KnowledgeBaseError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ):
        self.code = code
        self.message = message
        self.details = {} if details is None else details
        super().__init__(message)


class ConfigurationError(KnowledgeBaseError):
    pass
