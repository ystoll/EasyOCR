import pytest
import numpy as np
import os
import torch


from easyocr.utils import (BeamEntry, BeamState, addBeam, consecutive,
                           ctcBeamSearch, word_segmentation, simplify_label,
                           fast_simplify_label, CTCLabelConverter)


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
@pytest.mark.parametrize("test_case", [((), 0, 0, ()),
                                       ((), 46, 0, (46,)),
                                       ((46,), 0, 0, (46, 0)),
                                       ((46,), 74, 0, (46, 74)),
                                       ((46, 0), 74, 0, (46, 74)),
                                       ((46, 0), 74, 10, (46, 0, 74))])
def test_fast_simplify_label(test_case):
    assert fast_simplify_label(test_case[0],
                               test_case[1],
                               test_case[2]) == test_case[3]

@pytest.mark.parametrize("test_case", [ ((), 0, ()),
                                       ((), 46, ()),
                                       ((46,), 0, (46,))])
def test_simplify_label(test_case):
    assert simplify_label(test_case[0],
                          blankIdx=test_case[1]) == test_case[2]

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
@pytest.mark.parametrize("with_dict", [True, False])
def test_ctcBeamSearch(golden, load_fr_dict, load_mat_probs_mairie, with_dict):
    if with_dict:
        fr_dict = load_fr_dict
    else:
        fr_dict = []
    mat_probs = load_mat_probs_mairie  # shape: (28, 352)

    result = ctcBeamSearch(mat_probs,
                           golden["input"]["classes"],
                           golden["input"]["ignore_idx"],
                           lm=golden["input"]["lm"],
                           beamWidth=golden["input"]["beamWidth"],
                           dict_list=fr_dict)

    assert result == golden.out["output"]



class TestCTCLabelConverter():

    #@pytest.fixture(params=[{}, {"underscore": "_"}])
    @pytest.fixture(params=[{}])
    def get_CTCLabelConverter(self, request):
        character = (" !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                     "abcdefghijklmnopqrstuvwxyz{|}~ªÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæ"
                     "çèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćČčĎďĐđĒēĖėĘęĚěĞğĨĩĪīĮįİıĶķĹĺĻļĽľŁłŃńŅņ"
                     "ŇňŒœŔŕŘřŚśŞşŠšŤťŨũŪūŮůŲųŸŹźŻżŽžƏƠơƯưȘșȚțə̇ḌḍḶḷṀṁṂṃṄṅṆṇṬṭẠạẢảẤấẦầẨẩ"
                     "ẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤ"
                     "ụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ€")

        separator_list = request.param
        dict_list = {'fr': os.path.join(os.getcwd(), 'easyocr/dict/fr.txt')}
        converter = CTCLabelConverter(character, separator_list, dict_list)
        return converter

# TODO: CTCLabelConverter.encode is used only in trainer/.
    # def test_encode(self, get_CTCLabelConverter):
    #     converter = get_CTCLabelConverter
    #     return 0

    @pytest.mark.golden_test("data/test_easyocr_utils/test_CTCLabelConverter_decode_greedy.yaml")
    def test_decode_greedy(self, get_CTCLabelConverter, golden):
        converter = get_CTCLabelConverter
        assert converter.decode_greedy(np.array(golden["input"]["text_index"]),
                                       np.array(golden["input"]["length"])) == golden.out["output"]


    @pytest.mark.golden_test("data/test_easyocr_utils/test_CTCLabelConverter_decode_beamsearch.yaml")
    def test_decode_beamsearch(self, get_CTCLabelConverter, golden, load_mat_probs_mairie):
        converter = get_CTCLabelConverter
        mat_probs = load_mat_probs_mairie
        mat_probs = mat_probs[np.newaxis, :, :]

        assert converter.decode_beamsearch(mat_probs,
                                           np.array(golden["input"]["beamWidth"])) == golden.out["output"]

    @pytest.mark.golden_test("data/test_easyocr_utils/test_CTCLabelConverter_decode_beamsearch.yaml")
    def test_decode_wordbeamsearch(self, get_CTCLabelConverter, golden, load_mat_probs_mairie):
        converter = get_CTCLabelConverter
        mat_probs = load_mat_probs_mairie
        mat_probs = mat_probs[np.newaxis, :, :]

        assert converter.decode_beamsearch(mat_probs,
                                           np.array(golden["input"]["beamWidth"])) == golden.out["output"]



