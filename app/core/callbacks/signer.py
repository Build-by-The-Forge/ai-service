import hashlib
import hmac
import json
from typing import Any

from app.core.callbacks.events import CallbackEvent


def canonical_serialize(data: dict[str, Any] | CallbackEvent) -> bytes:
    """
    Serializes a callback event or dictionary into deterministic canonical JSON bytes.
    Keys are strictly sorted with compact separators to guarantee identical byte representations.
    """
    if isinstance(data, CallbackEvent):
        dict_data = data.to_dict()
    else:
        dict_data = data

    return json.dumps(dict_data, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CallbackSigner:
    """
    Cryptographic signer for outbound callbacks using HMAC-SHA256.
    Canonical signing format: {timestamp}.{canonical_json_payload}
    """

    def __init__(self, secret: str):
        self._secret = secret

    def sign(self, payload: bytes, timestamp: int) -> str:
        """
        Computes HMAC-SHA256 signature for the given canonical payload bytes and epoch timestamp.
        """
        canonical_material = f"{timestamp}.".encode() + payload
        return hmac.new(
            self._secret.encode("utf-8"),
            canonical_material,
            hashlib.sha256,
        ).hexdigest()

    def verify(self, signature: str, payload: bytes, timestamp: int) -> bool:
        """
        Constant-time verification of a callback signature.
        """
        expected_sig = self.sign(payload, timestamp)
        return hmac.compare_digest(signature, expected_sig)
