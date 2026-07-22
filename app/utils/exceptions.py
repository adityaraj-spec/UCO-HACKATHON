"""
app/utils/exceptions.py

Custom exception hierarchy for PhaseGuard Layer 2.

These exceptions are raised by repositories / services and translated into
appropriate HTTP responses by the API layer (see app/api/v1/endpoints/*.py
and the exception handlers registered in app/main.py).
"""


class PhaseGuardError(Exception):
    """Base class for all PhaseGuard application errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UserNotFoundError(PhaseGuardError):
    """Raised when a referenced user does not exist."""

    def __init__(self, user_id: str):
        super().__init__(f"User with id '{user_id}' was not found.")
        self.user_id = user_id


class UserAlreadyExistsError(PhaseGuardError):
    """Raised when attempting to create a user with a duplicate email."""

    def __init__(self, email: str):
        super().__init__(f"A user with email '{email}' already exists.")
        self.email = email


class VoiceprintNotFoundError(PhaseGuardError):
    """Raised when a user has not completed enrollment yet."""

    def __init__(self, user_id: str):
        super().__init__(
            f"No enrolled voiceprint found for user '{user_id}'. "
            "Please complete enrollment first."
        )
        self.user_id = user_id


class InvalidAudioFileError(PhaseGuardError):
    """Raised when an uploaded file is missing, empty, or unreadable as audio."""

    def __init__(self, filename: str, reason: str):
        super().__init__(f"Invalid audio file '{filename}': {reason}")
        self.filename = filename
        self.reason = reason


class InsufficientRecordingsError(PhaseGuardError):
    """Raised when an enrollment request does not include enough recordings."""

    def __init__(self, provided: int, required: int):
        super().__init__(
            f"Enrollment requires at least {required} valid recordings, "
            f"but only {provided} were provided/processed successfully."
        )
        self.provided = provided
        self.required = required


class EnrollmentRejectedError(PhaseGuardError):
    """Raised when enrollment audio fails liveness or quality checks."""

    def __init__(self, reason: str):
        super().__init__(f"Enrollment audio rejected: {reason}")
        self.reason = reason


class ModelLoadError(PhaseGuardError):
    """Raised when the ECAPA-TDNN model fails to load."""

    def __init__(self, reason: str):
        super().__init__(f"Failed to load speaker verification model: {reason}")
        self.reason = reason


class EmbeddingExtractionError(PhaseGuardError):
    """Raised when embedding extraction fails for a given audio file."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to extract speaker embedding from '{filename}': {reason}"
        )
        self.filename = filename
        self.reason = reason
