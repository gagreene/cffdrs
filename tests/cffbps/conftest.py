"""Run public FBP behavior against each available backend entry point."""

import pytest


@pytest.fixture(params=["python", "rust", "auto"])
def backend_fbp(request):
    """Return each backend's FBP class for the same behavioral and golden gates."""
    if request.param == "python":
        from cffdrs.cffbps.python import FBP
    elif request.param == "rust":
        from cffdrs.cffbps.rust import FBP
    else:
        from cffdrs.cffbps import FBP
    return FBP
