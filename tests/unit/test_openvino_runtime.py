"""
Unit tests for AI-2B.4.2 — OpenVINO Runtime Adapter.

Tests runtime initialization, device discovery, model validation, health checks,
and security constraints WITHOUT loading actual models or OpenVINO.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.documents import (
    ModelDescriptor,
    OpenVINODeviceNotFoundError,
    OpenVINOInitializationError,
    OpenVINOModelInvalidError,
    OpenVINOModelNotFoundError,
    OpenVINOPathValidationError,
    OpenVINORuntime,
)


class TestModelDescriptor:
    """Test generic model descriptor validation."""

    def test_model_descriptor_validates_xml_path_within_model_dir(self, tmp_path):
        """Model XML path must be within model_dir."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(tmp_path / "models" / "model.xml")
        Path(xml_path).touch()

        descriptor = ModelDescriptor(name="test", xml_path=xml_path)
        descriptor.validate_paths(model_dir)

    def test_model_descriptor_rejects_escaped_xml_path(self, tmp_path):
        """Reject XML paths that escape model_dir (../../secret.xml)."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(tmp_path / "secret.xml")
        Path(xml_path).touch()

        descriptor = ModelDescriptor(name="test", xml_path=xml_path)
        with pytest.raises(OpenVINOPathValidationError):
            descriptor.validate_paths(model_dir)

    def test_model_descriptor_requires_xml_to_exist(self, tmp_path):
        """Missing XML file raises ModelNotFoundError."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(tmp_path / "models" / "missing.xml")

        descriptor = ModelDescriptor(name="test", xml_path=xml_path)
        with pytest.raises(OpenVINOModelNotFoundError):
            descriptor.validate_paths(model_dir)

    def test_model_descriptor_validates_bin_path_within_model_dir(self, tmp_path):
        """Model BIN path must also be within model_dir."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(tmp_path / "models" / "model.xml")
        bin_path = str(tmp_path / "models" / "model.bin")
        Path(xml_path).touch()
        Path(bin_path).touch()

        descriptor = ModelDescriptor(name="test", xml_path=xml_path, bin_path=bin_path)
        descriptor.validate_paths(model_dir)

    def test_model_descriptor_rejects_escaped_bin_path(self, tmp_path):
        """Reject BIN paths that escape model_dir."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(tmp_path / "models" / "model.xml")
        bin_path = str(tmp_path / "secret.bin")
        Path(xml_path).touch()
        Path(bin_path).touch()

        descriptor = ModelDescriptor(name="test", xml_path=xml_path, bin_path=bin_path)
        with pytest.raises(OpenVINOPathValidationError):
            descriptor.validate_paths(model_dir)


class TestOpenVINORuntimeInitialization:
    """Test OpenVINO runtime initialization and device discovery."""

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_runtime_initializes_core_successfully(self, mock_core_factory, tmp_path):
        """Runtime successfully initializes OpenVINO Core."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=str(tmp_path), device="CPU")

        assert runtime.default_device == "CPU"
        assert runtime._available_devices == ["CPU"]
        assert runtime._core is not None

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_runtime_discovers_multiple_devices(self, mock_core_factory, tmp_path):
        """Runtime discovers available devices from OpenVINO Core."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU", "NPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=str(tmp_path), device="CPU")

        assert runtime._available_devices == ["CPU", "GPU", "NPU"]

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_runtime_rejects_unavailable_device(self, mock_core_factory, tmp_path):
        """Reject initialization if requested device is not available."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        with pytest.raises(OpenVINODeviceNotFoundError):
            OpenVINORuntime(model_dir=str(tmp_path), device="GPU")

    @patch("app.core.documents.openvino._resolve_openvino_core", side_effect=ImportError)
    def test_runtime_handles_openvino_not_installed(self, mock_core_factory, tmp_path):
        """Handle ImportError if OpenVINO is not installed."""
        with pytest.raises(OpenVINOInitializationError) as exc_info:
            OpenVINORuntime(model_dir=str(tmp_path))
        assert "OpenVINO is not installed" in str(exc_info.value)

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_runtime_handles_no_devices_found(self, mock_core_factory, tmp_path):
        """Fail initialization if no devices are detected."""
        mock_core = MagicMock()
        mock_core.available_devices = []
        mock_core_factory.return_value = lambda: mock_core

        with pytest.raises(OpenVINOInitializationError):
            OpenVINORuntime(model_dir=str(tmp_path))


class TestOpenVINORuntimeHealthCheck:
    """Test runtime health check and status reporting."""

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_health_check_returns_runtime_status(self, mock_core_factory, tmp_path):
        """Health check returns runtime status and capabilities."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=str(tmp_path), device="CPU")
        status = await runtime.health_check()

        assert status["healthy"] is True
        assert status["runtime"] == "openvino"
        assert status["default_device"] == "CPU"
        assert status["available_devices"] == ["CPU", "GPU"]
        assert status["loaded_models"] == []
        assert "version" in status

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_health_check_handles_core_exception(self, mock_core_factory, tmp_path):
        """Health check gracefully handles exceptions."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=str(tmp_path), device="CPU")
        runtime._core = None

        status = await runtime.health_check()

        assert status["healthy"] is False
        assert "reason" in status


class TestOpenVINORuntimeModelValidation:
    """Test model validation and security checks."""

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_validate_model_checks_device_availability(self, mock_core_factory, tmp_path):
        """Model validation checks that device is available."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(Path(model_dir) / "model.xml")
        Path(xml_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=xml_path, device="GPU")

        with pytest.raises(OpenVINODeviceNotFoundError):
            runtime.validate_model(descriptor)

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_validate_model_fails_on_invalid_ir(self, mock_core_factory, tmp_path):
        """Validation fails if OpenVINO cannot read the IR."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(Path(model_dir) / "model.xml")
        Path(xml_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core.read_model.side_effect = RuntimeError("Corrupted IR")
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=xml_path, device="CPU")

        with pytest.raises(OpenVINOModelInvalidError):
            runtime.validate_model(descriptor)

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_validate_model_blocks_path_traversal(self, mock_core_factory, tmp_path):
        """Model validation prevents path traversal attacks."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        secret_path = str(tmp_path / "secret.xml")
        Path(secret_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=secret_path, device="CPU")

        with pytest.raises(OpenVINOPathValidationError):
            runtime.validate_model(descriptor)


class TestOpenVINORuntimeCompilation:
    """Test model compilation (with mocked OpenVINO Core)."""

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_compile_model_validates_first(self, mock_core_factory, tmp_path):
        """Compilation runs validation before compiling."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(Path(model_dir) / "model.xml")
        Path(xml_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.return_value = MagicMock()
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=xml_path, device="CPU")

        key = await runtime.compile_model(descriptor)

        assert key == "test:CPU"
        mock_core.read_model.assert_called()
        mock_core.compile_model.assert_called()

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_compile_model_caches_result(self, mock_core_factory, tmp_path):
        """Compiled models are cached by key."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(Path(model_dir) / "model.xml")
        Path(xml_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.return_value = MagicMock()
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=xml_path, device="CPU")

        key1 = await runtime.compile_model(descriptor)
        key2 = await runtime.compile_model(descriptor)

        assert key1 == key2
        assert mock_core.compile_model.call_count == 1

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_compile_model_fails_on_invalid_path(self, mock_core_factory, tmp_path):
        """Compilation fails if model path is invalid."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        secret_path = str(tmp_path / "secret.xml")
        Path(secret_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=secret_path, device="CPU")

        with pytest.raises(OpenVINOPathValidationError):
            await runtime.compile_model(descriptor)

    @patch("app.core.documents.openvino._resolve_openvino_core")
    async def test_compile_model_handles_compilation_error(self, mock_core_factory, tmp_path):
        """Compilation errors are properly reported."""
        model_dir = str(tmp_path / "models")
        Path(model_dir).mkdir()
        xml_path = str(Path(model_dir) / "model.xml")
        Path(xml_path).touch()

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.side_effect = RuntimeError("Device not ready")
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=model_dir, device="CPU")
        descriptor = ModelDescriptor(name="test", xml_path=xml_path, device="CPU")

        from app.core.documents.openvino_errors import OpenVINOCompilationError

        with pytest.raises(OpenVINOCompilationError):
            await runtime.compile_model(descriptor)


class TestOpenVINORuntimeLifecycle:
    """Test runtime lifecycle management."""

    @patch("app.core.documents.openvino._resolve_openvino_core")
    def test_runtime_close_clears_compiled_models(self, mock_core_factory, tmp_path):
        """Close clears cached compiled models."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]
        mock_core_factory.return_value = lambda: mock_core

        runtime = OpenVINORuntime(model_dir=str(tmp_path), device="CPU")
        runtime._compiled_models["test:CPU"] = MagicMock()

        runtime.close()

        assert len(runtime._compiled_models) == 0
        assert runtime._core is None


# --- Phase 2 Step 1: no fake module, real cause kept ---------------------------------


class TestNoFakeOpenVINOModule:
    def test_importing_the_adapter_injects_nothing_when_openvino_is_absent(self, monkeypatch):
        import importlib
        import sys

        import app.core.documents.openvino as adapter

        monkeypatch.setitem(sys.modules, "openvino", None)  # simulate "not installed"
        try:
            reloaded = importlib.reload(adapter)
            assert sys.modules["openvino"] is None  # untouched: no fake module planted
            with pytest.raises(ImportError, match="OpenVINO is not installed") as exc:
                reloaded._resolve_openvino_core()
            assert isinstance(exc.value.__cause__, ModuleNotFoundError)
        finally:
            monkeypatch.undo()
            importlib.reload(adapter)

    def test_a_broken_installation_reports_its_real_cause(self, monkeypatch):
        import app.core.documents.openvino as adapter

        cause = OSError("[WinError 126] openvino_c.dll not found")

        def broken(name):
            raise cause

        monkeypatch.setattr(adapter, "Core", None)
        monkeypatch.setattr(adapter.importlib, "import_module", broken)
        with pytest.raises(ImportError, match="OpenVINO failed to load: OSError") as exc:
            adapter._resolve_openvino_core()
        assert exc.value.__cause__ is cause
