from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.callbacks.signer import CallbackSigner, canonical_serialize


def test_canonical_serialization_is_deterministic():
    data1 = {"b": 2, "a": 1, "nested": {"z": 26, "y": 25}}
    data2 = {"a": 1, "nested": {"y": 25, "z": 26}, "b": 2}

    bytes1 = canonical_serialize(data1)
    bytes2 = canonical_serialize(data2)

    assert bytes1 == bytes2
    assert bytes1 == b'{"a":1,"b":2,"nested":{"y":25,"z":26}}'


def test_callback_signature_is_deterministic():
    signer = CallbackSigner("test-secret-key-123")
    payload = b'{"event":"COMPLETED","job_id":"job_1"}'
    timestamp = 1788158400

    sig1 = signer.sign(payload, timestamp)
    sig2 = signer.sign(payload, timestamp)

    assert sig1 == sig2
    assert isinstance(sig1, str)
    assert len(sig1) == 64  # SHA-256 hex string


def test_callback_signature_changes_when_payload_changes():
    signer = CallbackSigner("test-secret-key-123")
    payload1 = b'{"event":"STARTED","job_id":"job_1"}'
    payload2 = b'{"event":"COMPLETED","job_id":"job_1"}'
    timestamp = 1788158400

    sig1 = signer.sign(payload1, timestamp)
    sig2 = signer.sign(payload2, timestamp)

    assert sig1 != sig2


def test_callback_signature_changes_when_timestamp_changes():
    signer = CallbackSigner("test-secret-key-123")
    payload = b'{"event":"COMPLETED","job_id":"job_1"}'

    sig1 = signer.sign(payload, 1788158400)
    sig2 = signer.sign(payload, 1788158401)

    assert sig1 != sig2


def test_signature_verification():
    signer = CallbackSigner("test-secret-key-123")
    event = CallbackEvent(
        job_id="job_test_01",
        asset_id="asset_test_01",
        tenant_id="school_1",
        run_number=1,
        attempt_count=1,
        external_task_id="task_001",
        correlation_id="corr_001",
        event_type=CallbackEventType.STARTED,
    )
    payload_bytes = canonical_serialize(event)
    timestamp = 1788158400

    signature = signer.sign(payload_bytes, timestamp)
    assert signer.verify(signature, payload_bytes, timestamp) is True

    # Tampered signature / tampered payload
    assert signer.verify("wrong_sig", payload_bytes, timestamp) is False
    assert signer.verify(signature, b'{"tampered":true}', timestamp) is False
    assert signer.verify(signature, payload_bytes, timestamp + 1) is False
