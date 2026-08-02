class KnowledgeBaseError(Exception):
    def __init__(self, code, message, details=None):
        self.code = code
        self.message = message
        self.details = {} if details is None else details
        super().__init__(message)


class ConfigurationError(KnowledgeBaseError):
    pass
