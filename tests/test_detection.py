import cv2
import numpy as np
import pytest
from pytest_golden.plugin import GoldenTestFixture

from easyocr import Reader
from tests.conftest import EasyOCROutput


@pytest.mark.golden_test("data/test_detection/test_readtext_simple.yml")
def test_readtext_simple(reader: Reader, golden: GoldenTestFixture) -> None:
    """Ensure the inference engine runs default text detection."""
    # Load the test image
    img = cv2.imread(golden["input"])
    assert isinstance(img, np.ndarray)

    # Run the engine
    output = reader.readtext(img)
    assert EasyOCROutput(output) == golden.out["output"]
