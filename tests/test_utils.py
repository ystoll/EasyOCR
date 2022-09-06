import pytest

from easyocr.utils import consecutive

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
