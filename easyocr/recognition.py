import importlib
import math
from collections import OrderedDict

import numpy as np
import torch
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from .utils import CTCLabelConverter


def custom_mean(prob):
    return prob.prod() ** (2.0 / np.sqrt(len(prob)))


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


class NormalizePAD(object):
    def __init__(self, max_size, pad_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.pad_type = pad_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c_num, h_img, w_img = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w_img] = img  # right pad
        if self.max_size[2] != w_img:  # add border Pad
            Pad_img[:, :, w_img:] = img[:, :, w_img - 1].unsqueeze(2).expand(c_num, h_img, self.max_size[2] - w_img)

        return Pad_img


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, "L")


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast=0.0):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w_img, h_img = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, "L")

            ratio = w_img / float(h_img)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


def recognizer_predict(
    model, converter, test_loader, batch_max_length, ignore_idx, char_group_idx, decoder="greedy", beam_width=5, device="cpu"
):
    del char_group_idx  # deleting for now unused variable char_group_idx.
    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            ######## filter ignore_char, rebalance
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            preds_prob[:, :, ignore_idx] = 0.0
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            if decoder == "greedy":
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
            elif decoder == "beamsearch":
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_beamsearch(k, beam_width=beam_width)
            elif decoder == "wordbeamsearch":
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_wordbeamsearch(k, beam_width=beam_width)

            preds_prob = preds_prob.cpu().detach().numpy()
            values = preds_prob.max(axis=2)
            indices = preds_prob.argmax(axis=2)
            preds_max_prob = []
            for curr_val, i in zip(values, indices):
                max_probs = curr_val[i != 0]
                if len(max_probs) > 0:
                    preds_max_prob.append(max_probs)
                else:
                    preds_max_prob.append(np.array([0]))

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = custom_mean(pred_max_prob)
                result.append([pred, confidence_score])

    return result


def get_recognizer(recog_network, network_params, character, separator_list, dict_list, model_path, device="cpu", quantize=True):

    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)

    if recog_network == "generation1":
        model_pkg = importlib.import_module("easyocr.model.model")
    elif recog_network == "generation2":
        model_pkg = importlib.import_module("easyocr.model.vgg_model")
    else:
        model_pkg = importlib.import_module(recog_network)
    model = model_pkg.Model(num_class=num_class, **network_params)

    if device == "cpu":
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model, converter


def get_text(
    character,
    imgH,
    imgW,
    recognizer,
    converter,
    image_list,
    ignore_char="",
    decoder="greedy",
    beam_width=5,
    batch_size=1,
    contrast_ths=0.1,
    adjust_contrast=0.5,
    filter_ths=0.003,
    workers=1,
    device="cpu",
):
    del filter_ths  # deleting for now unused variable filter_ths.
    batch_max_length = int(imgW / 10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try:
            ignore_idx.append(character.index(char) + 1)
        except:
            pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    align_collate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=int(workers), collate_fn=align_collate_normal, pin_memory=True
    )

    # predict first round
    result1 = recognizer_predict(
        recognizer, converter, test_loader, batch_max_length, ignore_idx, char_group_idx, decoder, beam_width, device=device
    )

    # predict second round
    low_confident_idx = [i for i, item in enumerate(result1) if item[1] < contrast_ths]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        align_collate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(workers),
            collate_fn=align_collate_contrast,
            pin_memory=True,
        )
        result2 = recognizer_predict(
            recognizer, converter, test_loader, batch_max_length, ignore_idx, char_group_idx, decoder, beam_width, device=device
        )

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1] > pred2[1]:
                result.append((box, pred1[0], pred1[1]))
            else:
                result.append((box, pred2[0], pred2[1]))
        else:
            result.append((box, pred1[0], pred1[1]))

    return result
