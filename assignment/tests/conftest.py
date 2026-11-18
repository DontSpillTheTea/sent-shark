import pytest

from assignment.rate_limit import rate_limiter
from assignment.services.judge_service import reset_cache


@pytest.fixture(autouse=True)
def _reset_in_memory_state():
    """Reset module-level caches and rate-limiter windows around every test."""
    reset_cache()
    rate_limiter.reset()
    yield
    reset_cache()
    rate_limiter.reset()
