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
    # Confidence intervals are rounded to the second decimal to avoid reproducibilities issues
    # between calls.
    output = list(map(lambda elem:(elem[0], elem[1], round(elem[2], 2)), output))
    assert EasyOCROutput(output) == golden.out["output"]
