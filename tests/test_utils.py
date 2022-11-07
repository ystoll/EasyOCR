import pytest

from easyocr.utils import consecutive, word_segmentation

# data: [28 29 40 41]
# mode: last
# result consecutive: [29, 41]

result_cons = consecutive(data= [28, 29, 40, 41, 50] , mode="first", stepsize=1)
print(f"result_cons: {result_cons}")

def test_consecutive():
    data = [28, 29, 40, 41]
    mode = "last"
    assert consecutive(data=data, mode=mode, stepsize=1) == [29, 41]

    data= [28, 29, 40, 41, 50]
    mode="first"
    assert consecutive(data=data, mode=mode, stepsize=1) == [28, 40, 50]

@pytest.mark.golden_test("golden_utils/test_word_segmentation.yaml")
@pytest.mark.parametrize("test", ["test_1", "test_2", "test_3"])
# @pytest.mark.parametrize("test", ["test_2"])
def test_word_segmentation(golden, test):
    assert word_segmentation(**golden[test]["input"]) == golden.out[test]["output"]