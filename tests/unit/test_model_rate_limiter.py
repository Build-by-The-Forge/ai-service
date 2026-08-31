from app.core.models.resilience import TokenBucketRateLimiter


def test_rate_limiter_allows_under_burst():
    limiter = TokenBucketRateLimiter(rate_per_second=10.0, burst_capacity=5)

    for _ in range(5):
        assert limiter.acquire("tenant_1", tokens=1) is True


def test_rate_limiter_rejects_when_exhausted():
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_capacity=2)

    assert limiter.acquire("tenant_1", tokens=1) is True
    assert limiter.acquire("tenant_1", tokens=1) is True
    # 3rd token exceeds burst capacity of 2
    assert limiter.acquire("tenant_1", tokens=1) is False

    # Different tenant has separate bucket
    assert limiter.acquire("tenant_2", tokens=1) is True
