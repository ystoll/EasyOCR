import pickle
from types import MappingProxyType

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


##### https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
class BeamEntry:
    "information about one single beam at specific time-step"

    def __init__(self):
        self.pr_total = 0  # blank and non-blank
        self.pr_non_blank = 0  # non-blank
        self.pr_blank = 0  # blank
        self.pr_text = 1  # LM score
        self.lm_applied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling


class BeamState:
    "information about the beams at specific time-step"

    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (entry_key, _) in self.entries.items():
            labeling_len = len(self.entries[entry_key].labeling)
            self.entries[entry_key].pr_text = self.entries[entry_key].pr_text ** (1.0 / (labeling_len if labeling_len else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_text)
        return [x.labeling for x in sorted_beams]

    def wordsearch(self, classes, ignore_idx, beam_width, dict_list):
        beams = [v for (_, v) in self.entries.items()]
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_text)[:beam_width]

        for index_candidate, candidate in enumerate(sorted_beams):
            idx_list = candidate.labeling
            text = ""
            for i, l in enumerate(idx_list):
                if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):  # removing repeated characters and blank.
                    text += classes[l]

            if index_candidate == 0:
                best_text = text
            if text in dict_list:
                print("found text: ", text)
                best_text = text
                break
            else:
                print("not in dict: ", text)
        return best_text


def apply_lm(parent_beam, child_beam, classes, lang_model):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lang_model and not child_beam.lm_applied:
        classe_1 = classes[parent_beam.labeling[-1] if parent_beam.labeling else classes.index(" ")]  # first char
        classe_2 = classes[child_beam.labeling[-1]]  # second char
        lang_model_factor = 0.01  # influence of language model
        bigram_prob = lang_model.getCharBigram(classe_1, classe_2) ** lang_model_factor  # probability of seeing first and second char next to each other
        child_beam.pr_text = parent_beam.pr_text * bigram_prob  # probability of char sequence
        child_beam.lm_applied = True  # only apply LM once per beam entry


def add_beam(beam_state, labeling):
    "add beam if it does not yet exist"
    if labeling not in beam_state.entries:
        beam_state.entries[labeling] = BeamEntry()


def ctc_beam_search(mat,
                    classes,
                    ignore_idx,
                    lang_model,
                    beam_width=25,
                    dict_list=None):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."
    if dict_list is None:
        dict_list = []

    # blank_idx = len(classes)
    blank_idx = 0
    max_t, max_c = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].pr_blank = 1
    last.entries[labeling].pr_total = 1

    # go over all time-steps
    for t in range(max_t):
        curr = BeamState()

        # get beam-labelings of best beams
        best_labelings = last.sort()[0:beam_width]

        # go over best beams
        for labeling in best_labelings:

            # probability of paths ending with a non-blank
            pr_non_blank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                pr_non_blank = last.entries[labeling].pr_non_blank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            pr_blank = (last.entries[labeling].pr_total) * mat[t, blank_idx]

            # add beam at current time-step if needed
            add_beam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank += pr_non_blank
            curr.entries[labeling].pr_blank += pr_blank
            curr.entries[labeling].pr_total += pr_blank + pr_non_blank
            curr.entries[labeling].pr_text = last.entries[
                labeling
            ].pr_text  # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lm_applied = True  # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(max_c - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_blank
                else:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_total

                # add beam at current time-step if needed
                add_beam(curr, new_labeling)

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank += pr_non_blank
                curr.entries[new_labeling].pr_total += pr_non_blank

                # apply LM
                # apply_lm(curr.entries[labeling], curr.entries[new_labeling], classes, lang_model)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    # sort by probability
    # best_labeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    # res = ''
    # for idx,l in enumerate(best_labeling):
    #    if l not in ignore_idx and (not (idx > 0 and best_labeling[idx - 1] == best_labeling[idx])):  # removing repeated characters and blank.
    #        res += classes[l]

    if dict_list == []:
        best_labeling = last.sort()[0]  # get most probable labeling
        res = ""
        for i, l in enumerate(best_labeling):
            if l not in ignore_idx and (
                not (i > 0 and best_labeling[i - 1] == best_labeling[i])
            ):  # removing repeated characters and blank.
                res += classes[l]
    else:
        res = last.wordsearch(classes, ignore_idx, beam_width, dict_list)

    return res


#####


def consecutive(data, mode="first", stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    group = [item for item in group if len(item) > 0]

    if mode == "first":
        result = [l[0] for l in group]
    elif mode == "last":
        result = [l[-1] for l in group]
    return result


def word_segmentation(mat,
                      separator_idx=MappingProxyType({"th": (1, 2), "en": (3, 4)}),
                      separator_idx_list=(1, 2, 3, 4)):
    result = []
    sep_list = []
    start_idx = 0
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0:
            mode = "first"
        else:
            mode = "last"
        a = consecutive(np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [[item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:  # start lang
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:  # end lang
                if sep_lang == lang:  # check if last entry if the same start lang
                    new_sep_pair = [lang, [sep_start_idx + 1, sep[0] - 1]]
                    if sep_start_idx > start_idx:
                        result.append(["", [start_idx, sep_start_idx - 1]])
                    start_idx = sep[0] + 1
                    result.append(new_sep_pair)
                else:  # reset
                    sep_lang = ""

    if start_idx <= len(mat) - 1:
        result.append(["", [start_idx, len(mat) - 1]])
    return result


class CTCLabelConverter(object):
    """Convert between text-label and text-index"""

    # def __init__(self, character, separator = []):
    def __init__(self, character, separator_list=None, dict_pathlist=None):
        if separator_list is None:
            separator_list = {}
        if dict_pathlist is None:
            dict_pathlist = {}
        # character (str): set of the possible characters.
        dict_character = list(character)

        # special_character = ['\xa2', '\xa3', '\xa4','\xa5']
        # self.separator_char = special_character[:len(separator)]

        self.dict = {}
        # for i, char in enumerate(self.separator_char + dict_character):
        for char_index, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = char_index + 1

        self.character = ["[blank]"] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        # self.character = ['[blank]']+ self.separator_char + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        self.separator_list = separator_list

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep

        self.ignore_idx = [0] + [i + 1 for i, item in enumerate(separator_char)]

        dict_list = {}
        for lang, dict_path in dict_pathlist.items():
            with open(dict_path, "rb") as input_file:
                word_count = pickle.load(input_file)
            dict_list[lang] = word_count
        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]

            char_list = []
            for i in range(l):
                if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                    # if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)

            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beam_width=5):
        texts = []

        for row_index in range(mat.shape[0]):
            text = ctc_beam_search(mat[row_index], self.character, self.ignore_idx, None, beam_width=beam_width)
            texts.append(text)
        return texts

    def decode_wordbeamsearch(self, mat, beam_width=5):
        texts = []
        argmax = np.argmax(mat, axis=2)
        for row_index in range(mat.shape[0]):
            words = word_segmentation(argmax[row_index])
            string = ""
            for word in words:
                matrix = mat[row_index, word[1][0] : word[1][1] + 1, :]
                if word[0] == "":
                    dict_list = []
                else:
                    dict_list = self.dict_list[word[0]]
                text = ctc_beam_search(matrix, self.character, self.ignore_idx, None, beam_width=beam_width, dict_list=dict_list)
                string += text
            texts.append(string)
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ["[GO]", "[s]"]  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for char_index, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = char_index

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append("[s]")
            text = [self.dict[char] for char in text]
            batch_text[i][1 : 1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, _ in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, value):
        count = value.data.numel()
        value = value.data.sum()
        self.n_count += count
        self.sum += value

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
