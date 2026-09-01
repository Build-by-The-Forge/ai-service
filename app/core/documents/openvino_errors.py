"""
OpenVINO runtime errors for AI-2B.4.2.

Follows the BUILD exception pattern: base class with error_code, provider/runtime context.
"""


class OpenVINOError(Exception):
    """Base exception for all OpenVINO runtime errors."""

    def __init__(
        self, message: str, error_code: str = "OPENVINO_ERROR", runtime_name: str = "openvino"
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.runtime_name = runtime_name


class OpenVINOInitializationError(OpenVINOError):
    """Raised when OpenVINO Core fails to initialize."""

    def __init__(self, message: str, runtime_name: str = "openvino"):
        super().__init__(
            f"OpenVINO Core initialization failed: {message}",
            error_code="OPENVINO_INIT_FAILED",
            runtime_name=runtime_name,
        )


class OpenVINODeviceNotFoundError(OpenVINOError):
    """Raised when a requested OpenVINO device is not available."""

    def __init__(self, requested_device: str, available_devices: list[str]):
        super().__init__(
            f"Device '{requested_device}' not available. Available: {available_devices}",
            error_code="OPENVINO_DEVICE_NOT_FOUND",
        )
        self.requested_device = requested_device
        self.available_devices = available_devices


class OpenVINOModelNotFoundError(OpenVINOError):
    """Raised when a model IR file cannot be found."""

    def __init__(self, xml_path: str, bin_path: str | None = None):
        paths = f"XML: {xml_path}"
        if bin_path:
            paths += f", BIN: {bin_path}"
        super().__init__(
            f"Model IR files not found: {paths}",
            error_code="OPENVINO_MODEL_NOT_FOUND",
        )
        self.xml_path = xml_path
        self.bin_path = bin_path


class OpenVINOModelInvalidError(OpenVINOError):
    """Raised when a model IR is invalid or cannot be read by OpenVINO."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Invalid model IR '{model_name}': {reason}",
            error_code="OPENVINO_MODEL_INVALID",
        )
        self.model_name = model_name
        self.reason = reason


class OpenVINOCompilationError(OpenVINOError):
    """Raised when model compilation for a device fails."""

    def __init__(self, model_name: str, device: str, reason: str):
        super().__init__(
            f"Failed to compile model '{model_name}' for device '{device}': {reason}",
            error_code="OPENVINO_COMPILATION_FAILED",
        )
        self.model_name = model_name
        self.device = device
        self.reason = reason


class OpenVINOPathValidationError(OpenVINOError):
    """Raised when a model path violates security constraints."""

    def __init__(self, attempted_path: str, reason: str):
        super().__init__(
            f"Model path validation failed: {reason}",
            error_code="OPENVINO_PATH_INVALID",
        )
        self.attempted_path = attempted_path
        self.reason = reason
