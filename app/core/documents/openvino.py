"""
OpenVINO Runtime Adapter for AI-2B.4.2.

Provides a BUILD-owned abstraction over OpenVINO Core for generic model loading and compilation.
Does NOT depend on PaddleOCR or any specific model architecture.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.documents.openvino_errors import (
    OpenVINOCompilationError,
    OpenVINODeviceNotFoundError,
    OpenVINOInitializationError,
    OpenVINOModelInvalidError,
    OpenVINOModelNotFoundError,
    OpenVINOPathValidationError,
)

Core: Any | None = None
__version__ = "unknown"

try:
    import openvino as _openvino_module  # type: ignore
except ModuleNotFoundError:
    _openvino_module = types.ModuleType("openvino")
    sys.modules.setdefault("openvino", _openvino_module)
else:
    sys.modules.setdefault("openvino", _openvino_module)

if not hasattr(sys.modules["openvino"], "Core"):
    setattr(sys.modules["openvino"], "Core", Core)
if not hasattr(sys.modules["openvino"], "__version__"):
    setattr(sys.modules["openvino"], "__version__", __version__)


def _resolve_openvino_core() -> Any:
    """Return the runtime Core class without requiring OpenVINO to be installed."""
    global Core
    if Core is not None:
        return Core

    try:
        module = importlib.import_module("openvino")
    except ModuleNotFoundError as exc:
        raise ImportError("OpenVINO is not installed. Install with: pip install openvino") from exc

    Core = getattr(module, "Core", None)
    if Core is None:
        raise ImportError("OpenVINO is not installed. Install with: pip install openvino")
    return Core


@dataclass(frozen=True)
class ModelDescriptor:
    """Generic model descriptor for OpenVINO IR (Intermediate Representation) models."""

    name: str
    xml_path: str
    bin_path: str | None = None
    device: str = "CPU"
    precision: str = "FP32"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_paths(self, model_dir: str) -> None:
        """
        Validate that model paths exist and are within the permitted model directory.

        Raises OpenVINOPathValidationError if paths escape model_dir or don't exist.
        """
        model_root = Path(model_dir).resolve()

        # Validate XML path
        xml_full = Path(self.xml_path).resolve()
        if not str(xml_full).startswith(str(model_root)):
            raise OpenVINOPathValidationError(
                self.xml_path,
                f"XML path escapes model directory: {model_dir}",
            )
        if not xml_full.exists():
            raise OpenVINOModelNotFoundError(self.xml_path, self.bin_path)

        # Validate BIN path if provided
        if self.bin_path:
            bin_full = Path(self.bin_path).resolve()
            if not str(bin_full).startswith(str(model_root)):
                raise OpenVINOPathValidationError(
                    self.bin_path,
                    f"BIN path escapes model directory: {model_dir}",
                )
            if not bin_full.exists():
                raise OpenVINOModelNotFoundError(self.xml_path, self.bin_path)


class OpenVINORuntime:
    """
    OpenVINO inference runtime adapter.

    Manages:
    - Core initialization and device discovery
    - Model validation and compilation
    - Lifecycle (init, health check, cleanup)

    Does NOT handle:
    - Actual inference (handled by adapters like PaddleOCRProvider)
    - Model-specific preprocessing/postprocessing
    - OCR-specific logic
    """

    def __init__(self, model_dir: str = "models/ocr", device: str = "CPU"):
        """
        Initialize OpenVINO runtime.

        Args:
            model_dir: Root directory for model IRs. All model paths must be within this dir.
            device: Default device for inference (CPU, GPU, NPU).

        Raises:
            OpenVINOInitializationError: if Core initialization fails.
            OpenVINODeviceNotFoundError: if requested device is not available.
        """
        self.model_dir = model_dir
        self.default_device = device.upper()
        self._core: Any = None
        self._available_devices: list[str] = []
        self._compiled_models: dict[str, Any] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Initialize OpenVINO Core and discover available devices."""
        try:
            core_factory = _resolve_openvino_core()
            self._core = core_factory()
            self._available_devices = self._core.available_devices
            if not self._available_devices:
                raise OpenVINOInitializationError("No devices detected by OpenVINO Core")

            if self.default_device not in self._available_devices:
                raise OpenVINODeviceNotFoundError(self.default_device, self._available_devices)
        except OpenVINODeviceNotFoundError:
            raise
        except ImportError as e:
            raise OpenVINOInitializationError(
                "OpenVINO is not installed. Install with: pip install openvino"
            ) from e
        except OpenVINOInitializationError:
            raise
        except Exception as e:
            raise OpenVINOInitializationError(str(e)) from e

    async def health_check(self) -> dict[str, Any]:
        """
        Check runtime health and capability.

        Returns:
            Dictionary with status, version, available devices, and loaded models.
        """
        try:
            if self._core is None:
                return {"healthy": False, "reason": "Core not initialized"}

            version = globals().get("__version__", "unknown")
            if version == "unknown":
                try:
                    version = importlib.import_module("openvino").__version__
                except Exception:
                    version = "unknown"

            return {
                "healthy": True,
                "runtime": "openvino",
                "version": version,
                "default_device": self.default_device,
                "available_devices": self._available_devices,
                "loaded_models": list(self._compiled_models.keys()),
                "model_directory": self.model_dir,
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "runtime": "openvino",
            }

    def validate_model(self, descriptor: ModelDescriptor) -> None:
        """
        Validate that a model can be loaded by OpenVINO.

        Checks:
        - Paths exist and are within model_dir
        - OpenVINO can read the IR
        - Device is available

        Raises:
            OpenVINOPathValidationError: if paths are invalid or escape model_dir
            OpenVINOModelNotFoundError: if files don't exist
            OpenVINOModelInvalidError: if OpenVINO cannot read the IR
            OpenVINODeviceNotFoundError: if device is not available
        """
        # Validate paths
        descriptor.validate_paths(self.model_dir)

        # Validate device
        device = descriptor.device.upper()
        if device not in self._available_devices:
            raise OpenVINODeviceNotFoundError(device, self._available_devices)

        # Try to read the model
        if self._core is None:
            raise OpenVINOInitializationError("Core not initialized")

        try:
            self._core.read_model(descriptor.xml_path)
        except Exception as e:
            raise OpenVINOModelInvalidError(descriptor.name, str(e)) from e

    async def compile_model(self, descriptor: ModelDescriptor) -> str:
        """
        Load and compile a model for the specified device.

        Args:
            descriptor: ModelDescriptor with model paths and device.

        Returns:
            A compilation key for later retrieval.

        Raises:
            OpenVINOModelInvalidError: if model is invalid
            OpenVINOCompilationError: if compilation fails
        """
        # Validate first
        self.validate_model(descriptor)

        if self._core is None:
            raise OpenVINOInitializationError("Core not initialized")

        device = descriptor.device.upper()
        compilation_key = f"{descriptor.name}:{device}"

        if compilation_key in self._compiled_models:
            return compilation_key

        try:
            model = self._core.read_model(descriptor.xml_path)
            compiled = self._core.compile_model(model, device)
            self._compiled_models[compilation_key] = compiled
            return compilation_key
        except Exception as e:
            raise OpenVINOCompilationError(descriptor.name, device, str(e)) from e

    def get_compiled_model(self, compilation_key: str) -> Any:
        """
        Retrieve a previously compiled model.

        Args:
            compilation_key: Key returned by compile_model.

        Returns:
            The compiled model object.

        Raises:
            KeyError: if the compilation key does not exist.
        """
        return self._compiled_models[compilation_key]

    def close(self) -> None:
        """Clean up runtime resources."""
        self._compiled_models.clear()
        self._core = None
