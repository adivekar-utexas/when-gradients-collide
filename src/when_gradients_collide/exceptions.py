"""Custom exception classes for WGC."""


class WGCError(Exception):
    """Base exception for all WGC errors."""


class PromptParsingError(WGCError):
    """Raised when JSON or LLM response parsing fails in the optimizer."""


class LLMBatchError(WGCError):
    """Raised when an LLM batch call fails."""


class InvalidBatchSizeError(WGCError):
    """Raised when batch_size configuration is invalid."""


class InvalidAlgorithmContextError(WGCError):
    """Raised when required keys are missing from the algorithm context."""
