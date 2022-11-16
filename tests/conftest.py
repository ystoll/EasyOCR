from typing import Any

import pytest
from pytest_golden.plugin import GoldenTestFixture
from pytest_golden import yaml

from easyocr import Reader


class EasyOCROutput(object):
    """An object compatible with YAML dumping for golden test fixtures."""

    value: str

    def __init__(self, value: Any) -> None:
        """Creates a new :class:`EasyOCROutput` instance."""
        self.value = str(value)

    def __eq__(self, __o: object) -> bool:
        return self.value == __o


yaml.add_representer(EasyOCROutput, lambda dumper, data: dumper.represent_scalar("!EasyOCROutput", data.value))
yaml.add_constructor('!EasyOCROutput', lambda loader, node: EasyOCROutput(node.value))


@pytest.fixture(scope="package")
def reader() -> Reader:
    """The default CRAFT inference engine."""
    # Initialize the inference engine using sane defaults
    engine = Reader(lang_list=["en"])

    return engine


@pytest.fixture(scope="package")
def reader_dbnet() -> Reader:
    """The DBNet inference engine."""
    # Initialize the inference engine using same defaults with DBNet
    # as the network
    engine = Reader(lang_list=["en"])

    return engine


@pytest.fixture
def compiled_golden_test(request: pytest.FixtureRequest) -> GoldenTestFixture:
    """Automatically compile a `GoldenTestFixture`."""
    print(request)
