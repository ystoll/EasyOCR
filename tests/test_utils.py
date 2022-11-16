import pytest
import numpy as np


from easyocr.utils import (BeamEntry, BeamState, addBeam, consecutive,
                           ctcBeamSearch, word_segmentation)


# Fixtures:
@pytest.fixture
def load_fr_dict():
    with open("easyocr/dict/fr.txt", "r") as dict_fr:
        return dict_fr.read().splitlines()

@pytest.fixture
def load_mat_probs_mairie():
    out_pickle = "tests/data/test_easyocr_utils/data/mat_proba_Mairie.csv"
    return np.genfromtxt(out_pickle, delimiter=',')


# Tests:
# "Normal"
def test_consecutive():
    data = [28, 29, 40, 41]
    mode = "last"
    assert consecutive(data=data, mode=mode, stepsize=1) == [29, 41]

    data= [28, 29, 40, 41, 50]
    mode="first"
    assert consecutive(data=data, mode=mode, stepsize=1) == [28, 40, 50]

# "Golden": inputs are read in Yaml files, except for dict or matrices
# which are either read from txt file (dicts) or csv files (matrices).

@pytest.mark.golden_test("data/test_easyocr_utils/test_word_segmentation.yaml")
@pytest.mark.parametrize("test", ["test_1", "test_2", "test_3"])
def test_word_segmentation(golden, test):
    assert word_segmentation(**golden[test]["input"]) == golden.out[test]["output"]


@pytest.mark.golden_test("data/test_easyocr_utils/test_ctcBeamSearch.yaml")
def test_ctcBeamSearch(golden, load_fr_dict, load_mat_probs_mairie):
    fr_dict = load_fr_dict
    mat_probs = load_mat_probs_mairie
    result = ctcBeamSearch(mat_probs,
                           golden["input"]["classes"],
                           golden["input"]["ignore_idx"],
                           lm=golden["input"]["ignore_idx"],
                           beamWidth=golden["input"]["beamWidth"],
                           dict_list=fr_dict)

    assert result == golden.out["output"]
