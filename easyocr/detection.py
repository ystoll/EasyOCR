from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from PIL import Image
# from torch.autograd import Variable

from .craft import CRAFT
from .craft_utils import adjustResultCoordinates, getDetBoxes
from .imgproc import normalizeMeanVariance, resize_aspect_ratio


def copy_state_dict(state_dict):
    """copy_state_dict _summary_

    Parameters
    ----------
    state_dict : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for param_name, param_value in state_dict.items():
        name = ".".join(param_name.split(".")[start_idx:])
        new_state_dict[name] = param_value
    return new_state_dict


def test_net(
    canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False
):
    """test_net _summary_

    Parameters
    ----------
    canvas_size : _type_
        _description_
    mag_ratio : _type_
        _description_
    net : _type_
        _description_
    image : _type_
        _description_
    text_threshold : _type_
        _description_
    link_threshold : _type_
        _description_
    low_text : _type_
        _description_
    poly : _type_
        _description_
    device : _type_
        _description_
    estimate_num_chars : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:  # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        del size_heatmap  # deleting unused size_heatmap variable.
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    curr_norm_img = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1)) for n_img in img_resized_list]
    curr_norm_img = torch.from_numpy(np.array(curr_norm_img))
    curr_norm_img = curr_norm_img.to(device)

    # forward pass
    with torch.no_grad():
        pred, feature = net(curr_norm_img)
        del feature  # deleting unused feature variable.

    boxes_list, polys_list = [], []
    for out in pred:
        # make score and link map.
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars
        )

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k_poly, poly in enumerate(polys):
            if estimate_num_chars:
                boxes[k_poly] = (boxes[k_poly], mapper[k_poly])
            if poly is None:
                polys[k_poly] = boxes[k_poly]

        boxes_list.append(boxes)
        polys_list.append(polys)


    return boxes_list, polys_list


def get_detector(trained_model, device="cpu", quantize=True, cudnn_benchmark=False):
    """get_detector _summary_

    Parameters
    ----------
    trained_model : _type_
        _description_
    device : str, optional
        _description_, by default "cpu"
    quantize : bool, optional
        _description_, by default True
    cudnn_benchmark : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    net = CRAFT()

    if device == "cpu":
        net.load_state_dict(copy_state_dict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copy_state_dict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net


def get_textbox(
    detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None
):
    """get_textbox _summary_

    Parameters
    ----------
    detector : _type_
        _description_
    image : _type_
        _description_
    canvas_size : _type_
        _description_
    mag_ratio : _type_
        _description_
    text_threshold : _type_
        _description_
    link_threshold : _type_
        _description_
    low_text : _type_
        _description_
    poly : _type_
        _description_
    device : _type_
        _description_
    optimal_num_chars : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(
        canvas_size, mag_ratio, detector, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars
    )
    del bboxes_list  # deleting bboxes_list unused variable.
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))] for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for box in polys:
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result
