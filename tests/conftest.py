import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LMK_ITEST") == "1":
        return
    skip = pytest.mark.skip(reason="set LMK_ITEST=1 to run integration tests (loads a real model)")
    for item in items:
        if "itest" in item.keywords:
            item.add_marker(skip)
