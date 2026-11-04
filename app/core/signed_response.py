"""Signed responses for routes that return generated or retrieved content
(M3-D provenance).

The body is serialised once, and those exact bytes are signed and sent, so
the backend verifies what it receives. The signature binds the response to
the verified request's id and the nonce the backend sent in its signed body.
"""

import time

from fastapi import Response
from pydantic import BaseModel

from app.core.config import settings
from app.core.service_signature_v1 import AI_SERVICE_ID, ResponseSigner, VerifiedCall


def signed_response(call: VerifiedCall, nonce: str, payload: BaseModel) -> Response:
    if not settings.AI_TO_BACKEND_SECRET:
        raise RuntimeError("AI_TO_BACKEND_SECRET is not configured")
    body = payload.model_dump_json().encode("utf-8")
    signer = ResponseSigner(sender=AI_SERVICE_ID, secret=settings.AI_TO_BACKEND_SECRET)
    headers = signer.headers(
        request_id=call.request_id, nonce=nonce, body=body, timestamp=int(time.time())
    )
    return Response(content=body, media_type="application/json", headers=headers)
