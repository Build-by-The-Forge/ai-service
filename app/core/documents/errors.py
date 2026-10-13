class DocumentError(Exception):
    """Base exception for all document processing and ingestion errors."""

    def __init__(
        self, message: str, error_code: str = "DOCUMENT_ERROR", tenant_id: str | None = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.tenant_id = tenant_id


class DocumentTenantMissingError(DocumentError):
    """Raised when a document is created or ingested without an authoritative tenant context."""

    def __init__(self, message: str = "Document requires an authoritative tenant_id"):
        super().__init__(message, error_code="DOCUMENT_TENANT_MISSING")


class CorruptDocumentError(DocumentError):
    """Raised when a document payload cannot be parsed or is corrupted."""

    def __init__(self, message: str, document_id: str | None = None, tenant_id: str | None = None):
        super().__init__(
            f"Corrupted document ({document_id or 'unknown'}): {message}",
            error_code="CORRUPT_DOCUMENT",
            tenant_id=tenant_id,
        )
        self.document_id = document_id


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when an uploaded file MIME or format is not supported."""

    def __init__(self, mime_type: str, tenant_id: str | None = None):
        super().__init__(
            f"Unsupported document MIME type: '{mime_type}'",
            error_code="UNSUPPORTED_DOCUMENT_TYPE",
            tenant_id=tenant_id,
        )
        self.mime_type = mime_type


class DocumentSizeLimitExceededError(DocumentError):
    """Raised when document byte size or page count exceeds tenant limits."""

    def __init__(
        self, message: str, max_limit: int, current_value: int, tenant_id: str | None = None
    ):
        super().__init__(message, error_code="DOCUMENT_SIZE_LIMIT_EXCEEDED", tenant_id=tenant_id)
        self.max_limit = max_limit
        self.current_value = current_value


class OCRExtractionError(DocumentError):
    """Raised when OCR extraction engine fails to process a page or image."""

    def __init__(self, message: str, page_number: int | None = None, tenant_id: str | None = None):
        super().__init__(
            f"OCR extraction failed on page {page_number or 'unknown'}: {message}",
            error_code="OCR_EXTRACTION_FAILED",
            tenant_id=tenant_id,
        )
        self.page_number = page_number


class InvalidBoundingBoxError(DocumentError):
    """Raised when an OCR bounding box has invalid coordinates."""

    def __init__(self, message: str):
        super().__init__(message, error_code="INVALID_BOUNDING_BOX")


class OCRNotImplementedError(DocumentError):
    """An OCR component was asked to do work it does not implement yet.

    Raised instead of returning an empty or fabricated result, so a caller can
    never mistake "nothing was recognised" for "recognition never ran".
    """

    def __init__(self, component: str, detail: str):
        super().__init__(
            f"{component} is not implemented: {detail}", error_code="OCR_NOT_IMPLEMENTED"
        )
        self.component = component


class OCRDependencyUnavailableError(DocumentError):
    """An OCR/document dependency (library or model) could not be loaded.

    Carries the original exception (also chained as ``__cause__``), so the
    real reason is never replaced by a generic "not installed". A missing
    package and a package that is installed but fails while importing (a
    native library, a broken dependency) are reported differently.
    """

    def __init__(
        self,
        dependency: str,
        cause: BaseException,
        install_hint: str | None = None,
        module: str | None = None,
    ):
        # "Missing" only when the package itself is absent; a missing module
        # *inside* an installed package is a load failure, reported in full.
        missing = isinstance(cause, ModuleNotFoundError) and (cause.name or "").split(".")[0] == (
            module or dependency
        )
        if missing:
            detail = "is not installed" + (f" ({install_hint})" if install_hint else "")
        else:
            detail = f"failed to load: {type(cause).__name__}: {cause}"
        super().__init__(f"{dependency} {detail}", error_code="OCR_DEPENDENCY_UNAVAILABLE")
        self.dependency = dependency
        self.cause = cause
        self.missing = missing
