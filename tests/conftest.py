"""
Pytest configuration and shared fixtures for the aws-model-deployment test suite.
"""

from hypothesis import settings, HealthCheck

# Default profile: used when running locally
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI profile: higher example count for thorough CI runs
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("default")
